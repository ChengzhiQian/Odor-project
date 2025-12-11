# src/models/simple_gnn.py
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(nn.Module):
    """
    图结构 + 节点特征 -> 图级表示 -> 气味多标签预测
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        """
        batch: PyG Batch, 要求包含:
          - batch.x          [N, in_dim]
          - batch.edge_index [2, E]
          - batch.batch      [N]
        """
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch_index)  # [B, hidden_dim]
        logits = self.lin(g)                  # [B, out_dim]
        return logits
