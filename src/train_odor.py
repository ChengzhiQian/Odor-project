# src/train_odor.py
from pathlib import Path
import copy

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


from src.datasets.mol_dataset import (
    load_molecule_dataset,
    smiles_to_pyg_graph_deepchem,
    inspect_data_deepchem,
    smiles_to_pyg_graph_chemprop,
    inspect_data_chemprop,
    smiles_to_pyg_graph_groverlite,
    inspect_data_groverlite,
)

from src.models.simple_gnn import SimpleGNN
from src.models.grover import GroverOnlyClassifier
from src.models.fusion_gnn import FusionGNN


# -------------------- 路径 & 配置 --------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 数据特征提取方式 (deepchem / chemprop / groverlite)
DATA_PROCESS_METHOD = "groverlite"

SMILES_TO_GRAPH = {
    "deepchem": smiles_to_pyg_graph_deepchem,
    "chemprop": smiles_to_pyg_graph_chemprop,
    "groverlite": smiles_to_pyg_graph_groverlite,
}

INSPECT_FN = {
    "deepchem": inspect_data_deepchem,
    "chemprop": inspect_data_chemprop,
    "groverlite": inspect_data_groverlite,
}

# 选择模型骨干："simple"（原 GNN）/ "grover" / "fusion"
BACKBONE = "grover"          # 改成 "grover" 或 "fusion" 就能跑另外两种


def get_featurizer(method: str):
    if method not in SMILES_TO_GRAPH:
        raise ValueError(f"Unknown DATA_PROCESS_METHOD: {method}")
    return SMILES_TO_GRAPH[method], INSPECT_FN[method]


# -------------------- metric 函数 --------------------
def odor_label_macro_f1(logits: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5,
                        eps: float = 1e-8) -> float:
    """
    按“每一种气味标签”计算 F1，然后对所有标签取平均（macro F1）。
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        B, L = labels.shape
        f1_list = []

        for j in range(L):
            y_true_j = labels[:, j]
            y_pred_j = preds[:, j]

            mask_j = ~torch.isnan(y_true_j)
            if mask_j.sum() == 0:
                continue

            y_true_j = y_true_j[mask_j]
            y_pred_j = y_pred_j[mask_j]

            tp = ((y_pred_j == 1) & (y_true_j == 1)).sum().float()
            fp = ((y_pred_j == 1) & (y_true_j == 0)).sum().float()
            fn = ((y_pred_j == 0) & (y_true_j == 1)).sum().float()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1_list.append(f1)

        if len(f1_list) == 0:
            return 0.0

        macro_f1 = torch.stack(f1_list).mean()
        return macro_f1.item()


def odor_sample_macro_f1(logits: torch.Tensor,
                         labels: torch.Tensor,
                         threshold: float = 0.5,
                         eps: float = 1e-8) -> float:
    """
    按“每一个分子”计算 F1，然后对所有样本取平均。
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        B, L = labels.shape
        f1_list = []

        for i in range(B):
            y_true_i = labels[i]
            y_pred_i = preds[i]

            mask_i = ~torch.isnan(y_true_i)
            if mask_i.sum() == 0:
                continue

            y_true_i = y_true_i[mask_i]
            y_pred_i = preds[i][mask_i]

            tp = ((y_pred_i == 1) & (y_true_i == 1)).sum().float()
            fp = ((y_pred_i == 1) & (y_true_i == 0)).sum().float()
            fn = ((y_pred_i == 0) & (y_true_i == 1)).sum().float()

            if tp + fp + fn == 0:
                f1 = torch.tensor(1.0, device=logits.device)
            else:
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2 * precision * recall / (precision + recall + eps)

            f1_list.append(f1)

        if len(f1_list) == 0:
            return 0.0

        macro_f1 = torch.stack(f1_list).mean()
        return macro_f1.item()

def odor_ranking_metrics_from_logits(
    logits_all: torch.Tensor,
    labels_all: torch.Tensor,
):
    """
    基于整轮(整個epoch)收集到的 logits/labels 计算:
      - macro-AUROC（按label平均，只统计可计算的label）
      - mAP (macro-AP)（按label平均；无正例/无标注记 0）
      - micro-PR-AUC（全元素 flatten）
      - micro-AUROC（全元素 flatten）
    支持 labels 含 NaN：会 mask 掉 NaN。
    """
    with torch.no_grad():
        y_true = labels_all.detach().cpu().numpy().astype(np.float64)   # (N, L)
        y_score = torch.sigmoid(logits_all).detach().cpu().numpy()      # (N, L)

    N, L = y_true.shape

    auc_list = []
    ap_list = []

    auc_valid = 0
    auc_invalid = 0
    ap_valid = 0
    ap_invalid = 0

    for j in range(L):
        col_true = y_true[:, j]
        col_score = y_score[:, j]

        mask = ~np.isnan(col_true)
        if mask.sum() == 0:
            # 没有标注
            ap_list.append(0.0)
            ap_invalid += 1
            auc_invalid += 1
            continue

        yt = (col_true[mask] > 0.5).astype(int)   # ✅ 强制二值
        ys = col_score[mask]

        # ---- AP: 即使全 0 也会返回 0（一般不报错），这里仍做 try 兜底
        try:
            ap_list.append(float(average_precision_score(yt, ys)))
            ap_valid += 1
        except Exception:
            ap_list.append(0.0)
            ap_invalid += 1

        # ---- AUROC: 必须同时存在 0 和 1
        if np.unique(yt).size < 2:
            auc_invalid += 1
        else:
            try:
                auc_list.append(float(roc_auc_score(yt, ys)))
                auc_valid += 1
            except Exception:
                auc_invalid += 1

    macro_auroc = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")
    mAP = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")
    macro_pr_auc = mAP

    # ---- micro metrics
    elem_mask = ~np.isnan(y_true)
    if elem_mask.any():
        yt_flat = (y_true[elem_mask] > 0.5).astype(int)
        ys_flat = y_score[elem_mask]

        micro_auroc = float(roc_auc_score(yt_flat, ys_flat)) if np.unique(yt_flat).size == 2 else float("nan")
        micro_pr_auc = float(average_precision_score(yt_flat, ys_flat)) if yt_flat.sum() > 0 else float("nan")
    else:
        micro_auroc = float("nan")
        micro_pr_auc = float("nan")

    return {
        "macro_auroc": macro_auroc,
        "mAP": mAP,
        "macro_pr_auc": macro_pr_auc,
        "micro_pr_auc": micro_pr_auc,
        "micro_auroc": micro_auroc,
        "num_labels_auc_valid": int(auc_valid),
        "num_labels_auc_invalid": int(auc_invalid),
        "num_labels_total": int(L),
        "num_labels_ap_valid": int(ap_valid),
        "num_labels_ap_invalid": int(ap_invalid),
    }



# -------------------- 主流程 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    smiles_to_graph_fn, inspect_fn = get_featurizer(DATA_PROCESS_METHOD)

    LABEL_COLS = None

    # 是否需要加载 GROVER fingerprint
    use_grover_fp = BACKBONE in {"grover", "fusion"}
    train_fp_path = str(DATA_PROCESSED_DIR / "train_grover_fp.npz") if use_grover_fp else None
    valid_fp_path = str(DATA_PROCESSED_DIR / "valid_grover_fp.npz") if use_grover_fp else None
    test_fp_path = str(DATA_PROCESSED_DIR / "test_grover_fp.npz") if use_grover_fp else None

    # ------- 加载数据 -------
    train_data = load_molecule_dataset(
        csv_path=str(DATA_PROCESSED_DIR / "train_dataset.csv"),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_graph_fn,
        grover_fp_path=train_fp_path,   # 如果是 simple 模型，这里就是 None
    )

    valid_data = load_molecule_dataset(
        csv_path=str(DATA_PROCESSED_DIR / "valid_dataset.csv"),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_graph_fn,
        grover_fp_path=valid_fp_path,
    )

    test_data = load_molecule_dataset(
        csv_path=str(DATA_PROCESSED_DIR / "test_dataset.csv"),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_graph_fn,
        grover_fp_path=test_fp_path,
    )

    print("train size:", len(train_data))
    print("valid size:", len(valid_data))
    print("test size:", len(test_data))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, exclude_keys=["lap_eigvec"])
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, exclude_keys=["lap_eigvec"])
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, exclude_keys=["lap_eigvec"])

    # ------- 定义模型 -------
    in_dim = train_data[0].x.size(-1)
    out_dim = train_data[0].y.size(-1)

    if BACKBONE == "simple":
        model = SimpleGNN(in_dim, hidden_dim=128, out_dim=out_dim).to(device)
    elif BACKBONE == "grover":
        fp_dim = train_data[0].grover_fp.size(-1)
        model = GroverOnlyClassifier(fp_dim=fp_dim, out_dim=out_dim).to(device)
    elif BACKBONE == "fusion":
        fp_dim = train_data[0].grover_fp.size(-1)
        model = FusionGNN(in_dim=in_dim, fp_dim=fp_dim,
                          hidden_dim=128, out_dim=out_dim).to(device)
    else:
        raise ValueError(f"Unknown BACKBONE: {BACKBONE}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1

    # ------- 训练 + 验证 -------
    for epoch in range(1, 101):
        # ===== train =====
        model.train()
        total_loss = 0.0
        total_label_f1 = 0.0
        total_sample_f1 = 0.0
        num_batches = 0

        train_logits_list = []  # ✅ 每个 epoch 重新初始化成 list
        train_labels_list = []

        for batch in train_loader:
            batch = batch.to(device)

            pred = model(batch)
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_label_f1 += odor_label_macro_f1(pred, batch.y)
            total_sample_f1 += odor_sample_macro_f1(pred, batch.y)
            num_batches += 1

            # ✅ 收集整轮数据（放 CPU，省显存）
            train_logits_list.append(pred.detach().cpu())
            train_labels_list.append(batch.y.detach().cpu())

        train_loss = total_loss / max(num_batches, 1)
        train_label_f1 = total_label_f1 / max(num_batches, 1)
        train_sample_f1 = total_sample_f1 / max(num_batches, 1)

        # ✅ epoch 结束后再 cat（cat 的结果用新名字，别覆盖 list）
        train_logits_all = torch.cat(train_logits_list, dim=0)
        train_labels_all = torch.cat(train_labels_list, dim=0)
        train_rank = odor_ranking_metrics_from_logits(train_logits_all, train_labels_all)

        # ===== valid =====
        model.eval()
        val_loss = 0.0
        val_label_f1 = 0.0
        val_sample_f1 = 0.0
        val_batches = 0

        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = loss_fn(pred, batch.y)

                val_loss += loss.item()
                val_label_f1 += odor_label_macro_f1(pred, batch.y)
                val_sample_f1 += odor_sample_macro_f1(pred, batch.y)
                val_batches += 1

                val_logits_list.append(pred.detach().cpu())
                val_labels_list.append(batch.y.detach().cpu())

        val_loss /= max(val_batches, 1)
        val_label_f1 /= max(val_batches, 1)
        val_sample_f1 /= max(val_batches, 1)

        val_logits_all = torch.cat(val_logits_list, dim=0)
        val_labels_all = torch.cat(val_labels_list, dim=0)
        val_rank = odor_ranking_metrics_from_logits(val_logits_all, val_labels_all)

        # ======= update best (based on val macroAUROC) =======
        val_score = val_rank["macro_auroc"]
        if not np.isnan(val_score) and val_score > best_val_auc:
            best_val_auc = float(val_score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, "
            f"train_labelF1={train_label_f1:.4f}, "
            f"train_sampleF1={train_sample_f1:.4f}, "
            f"train_macroAUROC={train_rank['macro_auroc']:.4f}, "
            f"train_mAP={train_rank['mAP']:.4f}, "
            f"train_microPRAUC={train_rank['micro_pr_auc']:.4f} | "
            f"val_loss={val_loss:.4f}, "
            f"val_labelF1={val_label_f1:.4f}, "
            f"val_sampleF1={val_sample_f1:.4f}, "
            f"val_macroAUROC={val_rank['macro_auroc']:.4f}, "
            f"val_mAP={val_rank['mAP']:.4f}, "
            f"val_microPRAUC={val_rank['micro_pr_auc']:.4f} | "
            f"val_AUC_valid={val_rank['num_labels_auc_valid']}/{val_rank['num_labels_total']}, "
            f"val_AP_valid={val_rank['num_labels_ap_valid']}/{val_rank['num_labels_total']}, "

        )

        print(f"Best epoch = {best_epoch}, best val macro-AUROC = {best_val_auc:.4f}")

        # load best
        if best_state is not None:
            model.load_state_dict(best_state)

        # eval on test
        model.eval()
        test_logits_list = []
        test_labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                test_logits_list.append(pred.detach().cpu())
                test_labels_list.append(batch.y.detach().cpu())

        test_logits_all = torch.cat(test_logits_list, dim=0)
        test_labels_all = torch.cat(test_labels_list, dim=0)
        test_rank = odor_ranking_metrics_from_logits(test_logits_all, test_labels_all)

        print(
            f"[TEST] macroAUROC={test_rank['macro_auroc']:.4f}, "
            f"mAP={test_rank['mAP']:.4f}, microPRAUC={test_rank['micro_pr_auc']:.4f}, "
            f"AUC_valid={test_rank.get('num_labels_auc_valid', '?')}/{test_rank.get('num_labels_total', '?')}\n"
        )


if __name__ == "__main__":
    main()
