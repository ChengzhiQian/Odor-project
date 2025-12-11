# src/train_odor.py
from pathlib import Path

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

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
DATA_PROCESS_METHOD = "chemprop"

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
BACKBONE = "fusion"          # 改成 "grover" 或 "fusion" 就能跑另外两种


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

    print("train size:", len(train_data))
    print("valid size:", len(valid_data))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)

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

    # ------- 训练 + 验证 -------
    for epoch in range(1, 101):
        # ===== train =====
        model.train()
        total_loss = 0.0
        total_label_f1 = 0.0
        total_sample_f1 = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            pred = model(batch)          # 统一接口：模型吃 batch，吐 logits
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_label_f1 += odor_label_macro_f1(pred, batch.y)
            total_sample_f1 += odor_sample_macro_f1(pred, batch.y)
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        train_label_f1 = total_label_f1 / max(num_batches, 1)
        train_sample_f1 = total_sample_f1 / max(num_batches, 1)

        # ===== valid =====
        model.eval()
        val_loss = 0.0
        val_label_f1 = 0.0
        val_sample_f1 = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = loss_fn(pred, batch.y)

                val_loss += loss.item()
                val_label_f1 += odor_label_macro_f1(pred, batch.y)
                val_sample_f1 += odor_sample_macro_f1(pred, batch.y)
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_label_f1 /= max(val_batches, 1)
        val_sample_f1 /= max(val_batches, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss = {train_loss:.4f}, "
            f"train_labelF1 = {train_label_f1:.4f}, "
            f"train_sampleF1 = {train_sample_f1:.4f} | "
            f"val_loss = {val_loss:.4f}, "
            f"val_labelF1 = {val_label_f1:.4f}, "
            f"val_sampleF1 = {val_sample_f1:.4f}"
        )


if __name__ == "__main__":
    main()
