# src/models/fusion_gnn.py
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class FusionGNN(nn.Module):
    """
    GNN(Graph) 表示 + GROVER fingerprint 融合，
    再做气味多标签预测。
    """

    def __init__(self,
                 in_dim: int,    # 节点特征维度 data.x.size(-1)
                 fp_dim: int,    # GROVER fingerprint 维度
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch):
        """
        需要 batch 里有:
          - x, edge_index, batch
          - grover_fp
        """
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # 图结构这一路
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        gnn_repr = global_mean_pool(x, batch_index)  # [B, hidden_dim]

        # GROVER fingerprint
        fp = batch.grover_fp  # [B, fp_dim]

        h = torch.cat([gnn_repr, fp], dim=-1)        # [B, hidden_dim + fp_dim]
        logits = self.mlp(h)
        return logits
