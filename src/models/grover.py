# src/models/grover.py
import torch
from torch import nn
from torch_geometric.data import Batch


class GroverOnlyClassifier(nn.Module):
    """
    只使用 GROVER fingerprint 做多标签分类的模型。

    期望输入：
      batch.grover_fp: [B, fp_dim] 或 [fp_dim]
    输出：
      logits: [B, out_dim]
    """

    def __init__(
        self,
        fp_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fp_dim = fp_dim

        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        batch: torch_geometric.data.Batch，里面有 batch.grover_fp

        返回：
          logits: [B, out_dim]
        """
        fp = batch.grover_fp

        # 常见几种情况统一处理成 [B, fp_dim]
        if fp.dim() == 1:
            # 单个样本，[fp_dim] -> [1, fp_dim]
            fp = fp.unsqueeze(0)
        elif fp.dim() == 2:
            # 正常情况：[B, fp_dim]，什么都不用改
            pass
        elif fp.dim() > 2:
            # 防止莫名其妙多出来维度，比如 [1, B, fp_dim] / [B, 1, fp_dim]
            # 统一假设第 0 维是 batch，其余全部展平
            B = fp.size(0)
            fp = fp.view(B, -1)

        # 这里强行检查一下，防止再出现 1x102400 这种情况
        if fp.size(1) != self.fp_dim:
            raise RuntimeError(
                f"[GroverOnlyClassifier] 期望每个样本特征维度 fp_dim={self.fp_dim}，"
                f"但当前 batch.grover_fp shape = {tuple(fp.shape)}"
            )

        logits = self.net(fp)   # [B, out_dim]
        return logits

class GroverFinetuneClassifier(nn.Module):
    """
    训练时在线跑 GROVER backbone（可插 adapter），再接分类头。
    期望 self.grover_backbone(batch) -> fp: [B, fp_dim]
    """

    def __init__(
        self,
        grover_backbone: nn.Module,
        fp_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        train_layernorm: bool = False,
    ):
        super().__init__()
        self.grover_backbone = grover_backbone
        self.fp_dim = fp_dim

        self.head = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        if freeze_backbone:
            for p in self.grover_backbone.parameters():
                p.requires_grad = False

        if train_layernorm:
            for n, p in self.grover_backbone.named_parameters():
                if "norm" in n.lower() or "layernorm" in n.lower():
                    p.requires_grad = True

    def forward(self, batch: Batch) -> torch.Tensor:
        fp = self.grover_backbone(batch)  # 必须返回 [B, fp_dim]
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(-1) != self.fp_dim:
            raise RuntimeError(f"[GroverFinetuneClassifier] backbone fp_dim mismatch: got {tuple(fp.shape)}")
        return self.head(fp)