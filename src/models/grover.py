# src/models/grover.py
import torch
from torch import nn
from torch_geometric.data import Batch


# class GroverOnlyClassifier(nn.Module):
#     """
#     只使用 GROVER fingerprint 做多标签分类的模型。
#
#     期望输入：
#       batch.grover_fp: [B, fp_dim] 或 [fp_dim]
#     输出：
#       logits: [B, out_dim]
#     """
#
#     def __init__(
#         self,
#         fp_dim: int,
#         out_dim: int,
#         hidden_dim: int = 512,
#         dropout: float = 0.2,
#     ):
#         super().__init__()
#         self.fp_dim = fp_dim
#
#         self.net = nn.Sequential(
#             nn.Linear(fp_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, out_dim),
#         )
#
#     def forward(self, batch: Batch) -> torch.Tensor:
#         """
#         batch: torch_geometric.data.Batch，里面有 batch.grover_fp
#
#         返回：
#           logits: [B, out_dim]
#         """
#         fp = batch.grover_fp
#
#         # 常见几种情况统一处理成 [B, fp_dim]
#         if fp.dim() == 1:
#             # 单个样本，[fp_dim] -> [1, fp_dim]
#             fp = fp.unsqueeze(0)
#         elif fp.dim() == 2:
#             # 正常情况：[B, fp_dim]，什么都不用改
#             pass
#         elif fp.dim() > 2:
#             # 防止莫名其妙多出来维度，比如 [1, B, fp_dim] / [B, 1, fp_dim]
#             # 统一假设第 0 维是 batch，其余全部展平
#             B = fp.size(0)
#             fp = fp.view(B, -1)
#
#         # 这里强行检查一下，防止再出现 1x102400 这种情况
#         if fp.size(1) != self.fp_dim:
#             raise RuntimeError(
#                 f"[GroverOnlyClassifier] 期望每个样本特征维度 fp_dim={self.fp_dim}，"
#                 f"但当前 batch.grover_fp shape = {tuple(fp.shape)}"
#             )
#
#         logits = self.net(fp)   # [B, out_dim]
#         return logits

class GroverOnlyClassifier(nn.Module):
    def __init__(self, fp_dim, out_dim, hidden_dim=512, dropout=0.2, use_layernorm=True):
        super().__init__()
        self.fp_dim = fp_dim
        self.out_dim = out_dim

        layers = [nn.Linear(fp_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))   # 稳定每个 batch 的表示，macro 常会更稳
        layers += [
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),           # 这里保留原来的线性输出
        ]
        self.net = nn.Sequential(*layers)

        # ✅ 关键：label-wise bias（每个标签一个可学习偏置）
        self.label_bias = nn.Parameter(torch.zeros(out_dim))

        # 可选：温度缩放（主要用于校准/稳定，不一定提升，但经常不伤）
        self.logit_scale = nn.Parameter(torch.ones(1))  # 等价于 1/temperature（简化版）

    @torch.no_grad()
    def init_bias_from_prior(self, p: torch.Tensor, eps: float = 1e-4):
        """
        p: [out_dim] 每个标签的正例比例（0~1）
        bias = log(p/(1-p))
        """
        p = p.clamp(eps, 1 - eps)
        b = torch.log(p / (1 - p))
        self.label_bias.copy_(b)

    def forward(self, batch: Batch) -> torch.Tensor:
        fp = batch.grover_fp
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        elif fp.dim() > 2:
            B = fp.size(0)
            fp = fp.view(B, -1)

        if fp.size(1) != self.fp_dim:
            raise RuntimeError(
                f"[GroverOnlyClassifier] 期望 fp_dim={self.fp_dim}，"
                f"但当前 batch.grover_fp shape = {tuple(fp.shape)}"
            )

        logits = self.net(fp)                       # [B, out_dim]
        logits = logits + self.label_bias           # ✅ 每个 label 单独平移
        logits = logits * self.logit_scale          # 可选缩放
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

class GroverAdapterClassifier(nn.Module):
    """
    end-to-end: smiles->graph->GroverFpGeneration(with adapter)->classifier
    只训练 adapter + 分类头，其它 Grover 参数冻结。
    """
    def __init__(self, grover_args, out_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.grover_fp = GroverFpGeneration(grover_args)  # 输出 [B, fp_dim]
        fp_dim = (4 * grover_args.hidden_size) if grover_args.fingerprint_source == "both" else (2 * grover_args.hidden_size)

        self.head = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # 关键：冻结 grover 里除了 adapter 之外的所有参数
        for n, p in self.grover_fp.named_parameters():
            p.requires_grad = ("atom_adapter" in n) or ("bond_adapter" in n) or ("adapter_scale" in n)

    def forward(self, batch: Batch) -> torch.Tensor:
        # 你现在的 batch 里已经有 graph_input / features_batch 吗？
        # 如果你的 dataset 里把 grover_repo 的 collator 输出存成了 batch.graph_input / batch.features_batch
        graph_batch = batch.graph_input
        features_batch = getattr(batch, "features_batch", [None])

        fp = self.grover_fp(graph_batch, features_batch)   # [B, fp_dim]
        return self.head(fp)