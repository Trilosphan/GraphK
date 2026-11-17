"""GKT tùy biến: lan truyền hai chiều trên đồ thị tiên quyết có hướng.

Dựa trên GKT (Nakagawa, Iwasawa & Matsuo, 2019), với các thay đổi theo dàn ý mục 3.3:
- mỗi khái niệm có một trạng thái ẩn riêng h_j;
- khi người học trả lời khái niệm i, nút i nhận thông điệp f_self, còn các nút lân cận nhận
  thông điệp theo chiều cạnh: f_fwd cho khái niệm phụ thuộc (i -> j), f_bwd cho khái niệm tiên quyết (j -> i);
- thông điệp qua cổng xóa–thêm (erase–add gate) rồi GRU; chỉ i và lân cận được cập nhật;
- lân cận được lưu dạng danh sách nén (CSR); mỗi bước chỉ tính trên các cạnh thật của những nút
  đang được trả lời, nên chi phí tỉ lệ với bậc thực tế thay vì với tổng số nút.

Xử lý nút cô lập (dàn ý mục 3.2), tham số `isolated`:
- 'self': nút cô lập chỉ tự cập nhật bằng f_self (tương đương vòng tự thân);
- 'root': thêm một nút gốc ảo nối tới mọi khái niệm không có tiên quyết (bậc vào = 0),
  biểu diễn năng lực chung của người học.
Chuẩn hóa trọng số thông điệp (`norm`): 'none' (mọi cạnh trọng số 1) hoặc 'row'
(chia cho số lân cận cùng chiều của nút được trả lời; bậc 0 không sinh phép chia).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn



def add_virtual_root(adj: sp.spmatrix) -> sp.csr_matrix:
    """Thêm nút gốc ảo (chỉ số n) với cạnh gốc -> mọi nút có bậc vào bằng 0."""
    a = sp.csr_matrix(adj, dtype=np.float32)
    n = a.shape[0]
    in_deg = np.asarray((a > 0).sum(axis=0)).ravel()
    roots = np.flatnonzero(in_deg == 0)
    out = sp.lil_matrix((n + 1, n + 1), dtype=np.float32)
    out[:n, :n] = a
    out[n, roots] = 1
    return out.tocsr()


def neighbor_lists(adj: sp.spmatrix, bidirectional: bool = True, norm: str = "row"):
    """Danh sách lân cận dạng nén (CSR): lân cận của nút i nằm ở nbr[ptr[i]:ptr[i+1]].

    Mỗi lân cận j xuất hiện đúng một lần, kèm hai trọng số thông điệp:
    w_fwd > 0 nếu có cạnh i -> j (j phụ thuộc i), w_bwd > 0 nếu có cạnh j -> i (j là tiên quyết của i).
    Đồ thị thống kê có cặp cạnh hai chiều (A -> B và B -> A); gộp về một mục để mỗi nút chỉ được
    cập nhật một lần trong một bước (index_put với chỉ số trùng cho kết quả không xác định).
    Không có ô đệm, nên chi phí mỗi bước tỉ lệ với bậc thực tế.
    """
    a = sp.csr_matrix(adj)
    n = a.shape[0]
    at = a.T.tocsr()
    ptr, nbr, wf, wb = [0], [], [], []
    for i in range(n):
        f = a.indices[a.indptr[i]:a.indptr[i + 1]].tolist()
        b = at.indices[at.indptr[i]:at.indptr[i + 1]].tolist() if bidirectional else []
        if norm == "row":
            cf, cb = (1.0 / len(f) if f else 0.0), (1.0 / len(b) if b else 0.0)
        else:
            cf = cb = 1.0
        fs, bs = set(f), set(b)
        for j in f + [j for j in b if j not in fs]:   # chiều xuôi trước, rồi các tiên quyết chưa có
            nbr.append(j)
            wf.append(cf if j in fs else 0.0)
            wb.append(cb if j in bs else 0.0)
        ptr.append(len(nbr))
    return (torch.tensor(ptr, dtype=torch.long), torch.tensor(nbr, dtype=torch.long),
            torch.tensor(wf, dtype=torch.float32), torch.tensor(wb, dtype=torch.float32))


def mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_out, d_out), nn.ReLU())


class EraseAddGate(nn.Module):
    """m' = m - w_c * e(m) * m + w_c * a(m), w_c là trọng số học được của từng khái niệm."""

    def __init__(self, d: int, n_nodes: int):
        super().__init__()
        self.erase = nn.Linear(d, d)
        self.add = nn.Linear(d, d)
        self.weight = nn.Parameter(torch.rand(n_nodes + 1) / (n_nodes + 1) ** 0.5)

    def forward(self, m: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        w = self.weight[nodes].unsqueeze(-1)
        return m - w * torch.sigmoid(self.erase(m)) * m + w * torch.tanh(self.add(m))


class GKT(nn.Module):
    def __init__(self, n_nodes: int, adj: sp.spmatrix, hidden: int = 32, emb: int = 32, dropout: float = 0.1,
                 bidirectional: bool = True, norm: str = "row", isolated: str = "self"):
        super().__init__()
        if isolated == "root":
            adj = add_virtual_root(adj)
        elif isolated != "self":
            raise ValueError(f"isolated phải là 'self' hoặc 'root', nhận {isolated!r}")
        self.n_nodes = n_nodes                 # số khái niệm thật (dùng để dự đoán)
        self.n_graph = adj.shape[0]            # số nút đồ thị (có thể thêm gốc ảo)
        ptr, nbr, wf, wb = neighbor_lists(adj, bidirectional, norm)
        self.register_buffer("nbr_ptr", ptr)
        self.register_buffer("nbr", nbr)
        self.register_buffer("nbr_wf", wf)
        self.register_buffer("nbr_wb", wb)
        self.hidden = hidden
        self.resp_emb = nn.Embedding(2 * n_nodes, emb)
        self.node_emb = nn.Embedding(self.n_graph, emb)
        self.h0 = nn.Parameter(torch.zeros(self.n_graph, hidden))
        self.f_self = mlp(hidden + emb, hidden, dropout)
        self.f_fwd = mlp(2 * (hidden + emb), hidden, dropout)
        self.f_bwd = mlp(2 * (hidden + emb), hidden, dropout)
        self.gate = EraseAddGate(hidden, self.n_graph)
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def _step(self, h: torch.Tensor, qt: torch.Tensor, rt: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Cập nhật trạng thái sau khi người học trả lời khái niệm qt với kết quả rt."""
        x = self.resp_emb(qt * 2 + rt)
        self_in = torch.cat([h[rows, qt], x], dim=-1)                          # (B, d+e)
        m_self = self.f_self(self_in)

        # Các cạnh của những nút đang được trả lời, trải phẳng: E_t = tổng bậc trong lô
        deg = self.nbr_ptr[1:][qt] - self.nbr_ptr[qt]
        e_rows = torch.repeat_interleave(rows, deg)                            # (E_t,)
        first = torch.repeat_interleave(self.nbr_ptr[qt] - (torch.cumsum(deg, 0) - deg), deg)
        pos = first + torch.arange(e_rows.numel(), device=qt.device)
        nb, wf, wb = self.nbr[pos], self.nbr_wf[pos], self.nbr_wb[pos]
        nb_in = torch.cat([self_in[e_rows], h[e_rows, nb], self.node_emb(nb)], dim=-1)
        m_nb = wf.unsqueeze(-1) * self.f_fwd(nb_in) + wb.unsqueeze(-1) * self.f_bwd(nb_in)

        upd_rows = torch.cat([rows, e_rows])
        upd_nodes = torch.cat([qt, nb])
        m_all = self.gate(torch.cat([m_self, m_nb]), upd_nodes)
        h_new = self.gru(m_all, h[upd_rows, upd_nodes])
        return h.index_put((upd_rows, upd_nodes), h_new)

    def forward(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """q, r: (B, T). Trả về logit (B, T-1): dự đoán r[:, t+1] từ q[:, :t+2] và r[:, :t+1]."""
        bsz, seqlen = q.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        rows = torch.arange(bsz, device=q.device)
        logits = []
        for t in range(seqlen - 1):
            h = self._step(h, q[:, t], r[:, t], rows)
            logits.append(self.out(h[rows, q[:, t + 1]]).squeeze(-1))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def trace(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Một chuỗi q, r (T,): xác suất trả lời đúng mọi khái niệm sau từng bước, kích thước (T, n_nodes)."""
        h = self.h0.unsqueeze(0).clone()
        rows = torch.zeros(1, dtype=torch.long, device=q.device)
        out = []
        for t in range(len(q)):
            h = self._step(h, q[t:t + 1], r[t:t + 1], rows)
            out.append(torch.sigmoid(self.out(h[0, :self.n_nodes])).squeeze(-1))
        return torch.stack(out)
