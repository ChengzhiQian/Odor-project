import copy
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from src.datasets.mol_dataset import (
    load_molecule_dataset,
    smiles_to_pyg_graph_deepchem,
    inspect_data_deepchem,
    smiles_to_pyg_graph_chemprop,
    inspect_data_chemprop,
    smiles_to_pyg_graph_groverlite,
    inspect_data_groverlite,
)

# ===================== 路径 & 配置 =====================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Data process method (deepchem/chemprop/groverlite)
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

# 是否启用 GraphCL 预训练
USE_GRAPHCL_PRETRAIN = True
GRAPHCL_EPOCHS = 50
FINETUNE_EPOCHS = 100


def get_featurizer(method: str):
    if method not in SMILES_TO_GRAPH:
        raise ValueError(f"Unknown DATA_PROCESS_METHOD: {method}")
    return SMILES_TO_GRAPH[method], INSPECT_FN[method]


# ===================== 评价指标：两个 macro F1 =====================

def odor_label_macro_f1(logits: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5,
                        eps: float = 1e-8) -> float:
    """
    按“每一种气味标签”计算 F1，然后对所有标签取平均（macro F1）。
    适合回答：整体上，每个气味预测得好不好？

    logits: [B, L]
    labels: [B, L] (0/1，若有 NaN 会自动忽略整列或整行)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        B, L = labels.shape
        f1_list = []

        for j in range(L):
            y_true_j = labels[:, j]
            y_pred_j = preds[:, j]

            # 忽略这一列里全是 NaN 的情况（如果有的话）
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
    适合回答：整体上，每个分子整体气味标签集合预测得好不好？

    logits: [B, L]
    labels: [B, L]
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
            y_pred_i = y_pred_i[mask_i]

            tp = ((y_pred_i == 1) & (y_true_i == 1)).sum().float()
            fp = ((y_pred_i == 1) & (y_true_i == 0)).sum().float()
            fn = ((y_pred_i == 0) & (y_true_i == 1)).sum().float()

            if tp + fp + fn == 0:
                # 这个分子没有任何正标签，预测全 0 的话我们可以认为 F1 = 1
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


# ===================== 图增强（GraphCL 风格） =====================

def drop_edge(data: Data, drop_prob: float = 0.2) -> Data:
    """
    随机删边：对整个 batch 图统一处理。
    """
    new_data = copy.deepcopy(data)
    edge_index = new_data.edge_index
    num_edges = edge_index.size(1)

    if num_edges == 0:
        return new_data

    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    if mask.sum() == 0:
        # 如果全删掉了，为了避免图崩掉，保留原始边
        return new_data

    new_data.edge_index = edge_index[:, mask]

    if hasattr(new_data, "edge_attr") and new_data.edge_attr is not None:
        new_data.edge_attr = new_data.edge_attr[mask]

    return new_data


def mask_node_features(data: Data, mask_prob: float = 0.2) -> Data:
    """
    随机把部分节点特征置 0。
    """
    new_data = copy.deepcopy(data)
    x = new_data.x
    num_nodes = x.size(0)

    if num_nodes == 0:
        return new_data

    mask = torch.rand(num_nodes, device=x.device) < mask_prob
    if mask.any():
        x[mask] = 0.0
    new_data.x = x
    return new_data


def graph_augment(data: Data) -> Data:
    """
    GraphCL 里的一种简单增强组合：
    1) 随机删边
    2) 随机 mask 节点特征
    """
    aug = drop_edge(data, drop_prob=0.2)
    aug = mask_node_features(aug, mask_prob=0.2)
    return aug


# ===================== GNN Encoder & 下游分类头 =====================

class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)  # [B, hidden_dim]
        return g


class OdorClassifier(nn.Module):
    def __init__(self, encoder: GNNEncoder, out_dim: int):
        super().__init__()
        self.encoder = encoder
        self.lin = nn.Linear(encoder.hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        g = self.encoder(x, edge_index, batch)
        logits = self.lin(g)  # [B, L]
        return logits


# ===================== GraphCL 模型 & 对比损失 =====================

class GraphCLModel(nn.Module):
    """
    GraphCL 风格：在 GNNEncoder 上叠一个投影头，输出对比学习的表示。
    """

    def __init__(self, encoder: GNNEncoder, proj_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder.hidden_dim, proj_dim),
        )

    def encode(self, data: Data):
        """
        输入一个 batch 的图（Data / Batch），输出 L2-normalized 的图表示。
        """
        h = self.encoder(data.x, data.edge_index, data.batch)  # [B, hidden_dim]
        z = self.projector(h)                                  # [B, proj_dim]
        z = F.normalize(z, dim=-1)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    一个简化版的 NT-Xent / InfoNCE：
    - 正样本：z1[i] <-> z2[i]
    - 对每个 z1[i]，负样本是 z2[j] (j != i)
    - 对每个 z2[i]，负样本是 z1[j] (j != i)
    """
    B = z1.size(0)
    # 归一化在外面已经做了，这里是内积相似度
    sim_12 = torch.mm(z1, z2.t()) / tau  # [B, B]
    sim_21 = sim_12.t()                  # [B, B]

    labels = torch.arange(B, device=z1.device)

    loss_1 = F.cross_entropy(sim_12, labels)
    loss_2 = F.cross_entropy(sim_21, labels)

    loss = 0.5 * (loss_1 + loss_2)
    return loss


# ===================== GraphCL 自监督预训练 =====================

def pretrain_graphcl(encoder: GNNEncoder,
                     train_loader: DataLoader,
                     device: torch.device,
                     epochs: int = 50,
                     lr: float = 1e-3) -> GNNEncoder:
    """
    使用 GraphCL 风格对比学习，对 encoder 做自监督预训练。
    """
    ssl_model = GraphCLModel(encoder).to(device)
    optimizer = optim.Adam(ssl_model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # 生成两个增强视图
            batch1 = graph_augment(batch)
            batch2 = graph_augment(batch)

            z1 = ssl_model.encode(batch1)
            z2 = ssl_model.encode(batch2)

            loss = nt_xent_loss(z1, z2, tau=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[GraphCL pretrain] Epoch {epoch:03d} | loss = {avg_loss:.4f}")

    # encoder 的参数已经在 ssl_model 里被更新，返回引用以便后续微调使用
    return ssl_model.encoder


# ===================== 主流程：加载数据 + 预训练 + 微调 =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    smiles_to_graph_fn, inspect_fn = get_featurizer(DATA_PROCESS_METHOD)

    # 1. 定义 label 列（None 表示在 load_molecule_dataset 里用默认配置）
    LABEL_COLS = None

    train_data = load_molecule_dataset(
        csv_path=str(DATA_PROCESSED_DIR / "train_dataset.csv"),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_graph_fn,
    )

    valid_data = load_molecule_dataset(
        csv_path=str(DATA_PROCESSED_DIR / "valid_dataset.csv"),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_graph_fn,
    )

    print("train size:", len(train_data))
    print("valid size:", len(valid_data))

    # 看一眼图结构（需要的话）
    # first_graph = train_data[0]
    # inspect_fn(first_graph, max_edges=40)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)

    in_dim = train_data[0].x.size(-1)
    out_dim = train_data[0].y.size(-1)
    hidden_dim = 128

    # 2. 构造 GNN encoder
    encoder = GNNEncoder(in_dim, hidden_dim).to(device)

    # 3. GraphCL 自监督预训练（可选）
    if USE_GRAPHCL_PRETRAIN:
        print("==== Start GraphCL pretraining ====")
        encoder = pretrain_graphcl(
            encoder=encoder,
            train_loader=train_loader,
            device=device,
            epochs=GRAPHCL_EPOCHS,
            lr=1e-3,
        )
        print("==== Finish GraphCL pretraining ====")
    else:
        print("GraphCL pretraining is disabled, directly training supervised model.")

    # 4. 下游有监督微调：气味多标签预测
    model = OdorClassifier(encoder, out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        # ===== train =====
        model.train()
        total_loss = 0.0
        total_label_f1 = 0.0
        total_sample_f1 = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.batch)
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
                pred = model(batch.x, batch.edge_index, batch.batch)
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
            f"train_labelF1 = {train_label_f1:.4f}, train_sampleF1 = {train_sample_f1:.4f} | "
            f"val_loss = {val_loss:.4f}, "
            f"val_labelF1 = {val_label_f1:.4f}, val_sampleF1 = {val_sample_f1:.4f}"
        )


if __name__ == "__main__":
    main()
