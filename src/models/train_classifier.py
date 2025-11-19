from torch_geometric.loader import DataLoader
from torch import nn, optim
import torch
from pathlib import Path

from src.datasets.mol_dataset import (
    load_molecule_dataset,
    smiles_to_pyg_graph_deepchem,
    inspect_data_deepchem,
    smiles_to_pyg_graph_chemprop,
    inspect_data_chemprop,
    smiles_to_pyg_graph_groverlite,
    inspect_data_groverlite,
)

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

def get_featurizer(method: str):
    if method not in SMILES_TO_GRAPH:
        raise ValueError(f"Unknown DATA_PROCESS_METHOD: {method}")
    return SMILES_TO_GRAPH[method], INSPECT_FN[method]

# Here, we use a simple GNN just to ensure that it works
def odor_label_macro_f1(logits: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5,
                        eps: float = 1e-8) -> float:
    """
    按“每一种气味标签”计算 F1，然后对所有标签取平均（macro F1）。
    适合回答：整体上，每个气味预测得好不好？

    logits: [B, L]
    labels: [B, L]  (0/1，若有 NaN 会自动忽略整列或整行)
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
                # 或者直接跳过，这里选择给 1，避免影响平均值
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    smiles_to_graph_fn, inspect_fn = get_featurizer(DATA_PROCESS_METHOD)

    # 1. Define label columns
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

    first_graph = train_data[0]
    # inspect_fn(first_graph, max_edges=40)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)

    # 2. Quickly build a small GNN model (replace it this with GAT / GraphTransformer later)
    from torch_geometric.nn import GCNConv, global_mean_pool

    class SimpleGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            out = self.lin(x)
            return out

    in_dim = train_data[0].x.size(-1)
    out_dim = train_data[0].y.size(-1)
    model = SimpleGNN(in_dim, 128, out_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. train + validation
    for epoch in range(1, 101):
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
