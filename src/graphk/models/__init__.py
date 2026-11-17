"""Các mô hình truy vết tri thức. Mọi mô hình nhận (q, r) kích thước (B, T) và trả về logit (B, T-1)
cho việc dự đoán r[:, 1:]."""

from __future__ import annotations

import torch
from torch import nn

from graphk.models.dkt import DKT
from graphk.models.gkt import GKT


class AKTWrapper(nn.Module):
    """Bọc AKT của pyKT về cùng giao diện: trả về logit cho r[:, 1:]."""

    def __init__(self, n_nodes: int, d_model: int = 256, n_blocks: int = 4, dropout: float = 0.2,
                 d_ff: int = 256, n_heads: int = 8, final_fc_dim: int = 512):
        super().__init__()
        from graphk.models.akt_pykt import AKT
        self.akt = AKT(n_question=n_nodes, n_pid=0, d_model=d_model, n_blocks=n_blocks, dropout=dropout,
                       d_ff=d_ff, num_attn_heads=n_heads, final_fc_dim=final_fc_dim)

    def forward(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        preds, _ = self.akt(q, r)          # xác suất; vị trí t chỉ thấy r[:, :t]
        p = preds[:, 1:].clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)


def build_model(name: str, n_nodes: int, adj=None, **kwargs) -> nn.Module:
    if name == "gkt":
        if adj is None:
            raise ValueError("GKT cần ma trận kề")
        return GKT(n_nodes, adj, **kwargs)
    if name == "dkt":
        return DKT(n_nodes, **kwargs)
    if name == "akt":
        return AKTWrapper(n_nodes, **kwargs)
    raise ValueError(f"Mô hình không hỗ trợ: {name}")
