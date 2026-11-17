"""DKT (Piech và cộng sự, 2015): LSTM trên chuỗi (khái niệm, kết quả), không dùng đồ thị."""

from __future__ import annotations

import torch
from torch import nn


class DKT(nn.Module):
    def __init__(self, n_nodes: int, emb: int = 64, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.inter_emb = nn.Embedding(2 * n_nodes, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_nodes)

    def forward(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Trả về logit (B, T-1): dự đoán r[:, t+1] từ tương tác 0..t."""
        h, _ = self.lstm(self.inter_emb(q[:, :-1] * 2 + r[:, :-1]))
        y = self.out(self.dropout(h))                                   # (B, T-1, n_nodes)
        return torch.gather(y, 2, q[:, 1:].unsqueeze(-1)).squeeze(-1)
