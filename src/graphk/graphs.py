"""Đồ thị đối chứng cho GKT: đồ thị thống kê, đồ thị ngẫu nhiên, đồ thị đầy đủ; tập khái niệm giữ lại.

Đồ thị thống kê (GKT-Stat) được xây dựng từ hành vi làm bài:
- mỗi người học u, mỗi khái niệm c đã làm: m[u, c] = 1 nếu tỉ lệ đúng của u trên c >= ngưỡng thành thạo;
- với cặp (A, B), chỉ xét người học đã làm cả A và B;
- điểm có hướng A -> B là giá trị gia tăng (added value):
      AV(A -> B) = P(thành thạo B | thành thạo A) - P(thành thạo B)
  tức mức tăng xác suất thành thạo B khi đã biết người học thành thạo A;
- giữ các cặp có đủ hỗ trợ (số người học thành thạo A và đã làm B >= min_support);
- chọn đúng n_edges cạnh (bằng số cạnh của đồ thị chuyên gia): trước hết mỗi khái niệm nhận
  cạnh vào có AV lớn nhất, phần còn lại lấy theo AV toàn cục (xem top_k_edges).

Chống rò rỉ: đồ thị chỉ được tính trên người học thuộc tập huấn luyện của từng phần,
và không có cạnh nào chạm vào tập khái niệm giữ lại.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graphk.preprocess import build_adjacency


# ---------------------------------------------------------------- đọc dữ liệu đã tiền xử lý

def load_processed(proc_dir: str | Path) -> dict:
    proc = Path(proc_dir)
    seqs = dict(np.load(proc / "sequences.npz"))
    return {
        "nodes": pd.read_csv(proc / "nodes.csv"),
        "adj_expert": sp.load_npz(proc / "adj_expert.npz").tocsr(),
        "sequences": seqs,
        "splits": pd.read_csv(proc / "user_split.csv", dtype={"split": str}),
    }


def interactions_of_users(seqs: dict[str, np.ndarray], users) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (user, node, correct) của mọi tương tác thuộc các người học cho trước."""
    lengths = np.diff(seqs["offsets"])
    inter_user = np.repeat(seqs["seq_user"], lengths)
    keep = np.isin(inter_user, np.asarray(users))
    return inter_user[keep], seqs["questions"][keep].astype(np.int64), seqs["responses"][keep].astype(np.int64)


def fold_users(splits: pd.DataFrame, fold: int | None) -> np.ndarray:
    """Người học dùng để xây dựng đồ thị: phần k -> các phần huấn luyện (khác k, khác 'test');
    fold=None -> toàn bộ người học không thuộc tập kiểm tra (dùng khi đánh giá cuối trên tập kiểm tra)."""
    mask = splits["split"] != "test"
    if fold is not None:
        mask &= splits["split"] != str(fold)
    return splits.loc[mask, "user_id"].to_numpy()


# ---------------------------------------------------------------- tập khái niệm giữ lại

def choose_heldout_concepts(nodes: pd.DataFrame, frac: float = 0.15, seed: int = 42) -> np.ndarray:
    """Chọn ngẫu nhiên frac khái niệm trong số khái niệm có tương tác."""
    candidates = nodes.loc[nodes["n_interactions"] > 0, "node_id"].to_numpy()
    rng = np.random.default_rng(seed)
    k = int(round(len(candidates) * frac))
    return np.sort(rng.choice(candidates, size=k, replace=False))


# ---------------------------------------------------------------- đồ thị thống kê

def mastery_matrices(user, node, correct, n_nodes: int, threshold: float = 0.5) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Ma trận người học × khái niệm: att[u, c] = đã làm; mas[u, c] = tỉ lệ đúng >= threshold."""
    _, u_idx = np.unique(user, return_inverse=True)
    key = u_idx.astype(np.int64) * n_nodes + node
    uniq, inv = np.unique(key, return_inverse=True)
    rate = np.bincount(inv, weights=correct) / np.bincount(inv)
    rows, cols = uniq // n_nodes, uniq % n_nodes
    shape = (int(u_idx.max()) + 1 if len(u_idx) else 0, n_nodes)
    att = sp.csr_matrix((np.ones(len(uniq), dtype=np.float32), (rows, cols)), shape=shape)
    m = rate >= threshold
    mas = sp.csr_matrix((np.ones(int(m.sum()), dtype=np.float32), (rows[m], cols[m])), shape=shape)
    return att, mas


def added_value_scores(att: sp.csr_matrix, mas: sp.csr_matrix, min_support: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """AV[A, B] = P(mas B | mas A, đã làm B) - P(mas B | đã làm A và B); NaN nếu không đủ hỗ trợ.

    Trả về (AV, support) với support[A, B] = số người học thành thạo A và đã làm B.
    """
    mm = (mas.T @ mas).toarray()   # thành thạo A và thành thạo B
    ma = (mas.T @ att).toarray()   # thành thạo A, đã làm B
    am = (att.T @ mas).toarray()   # đã làm A, thành thạo B
    aa = (att.T @ att).toarray()   # đã làm cả A và B
    with np.errstate(divide="ignore", invalid="ignore"):
        av = mm / ma - am / aa
    av[(ma < min_support) | (aa == 0)] = np.nan
    np.fill_diagonal(av, np.nan)
    return av, ma


def top_k_edges(scores: np.ndarray, k: int, exclude_nodes=(), per_node: int = 1) -> list[tuple[int, int]]:
    """Chọn k cặp (A, B) theo điểm, bỏ NaN, điểm <= 0 và mọi cặp chạm vào exclude_nodes.

    Bước 1: mỗi nút đích B nhận per_node cạnh vào có điểm cao nhất (nếu có), để đồ thị phủ
    rộng như đồ thị chuyên gia thay vì dồn cạnh vào vài nút. Bước 2: lấy thêm theo điểm toàn cục
    đến khi đủ k. per_node=0 chỉ dùng bước 2.
    """
    s = scores.copy()
    ex = np.asarray(list(exclude_nodes), dtype=np.int64)
    if len(ex):
        s[ex, :] = np.nan
        s[:, ex] = np.nan
    s[~(s > 0)] = np.nan  # NaN và điểm không dương đều bị loại
    n = s.shape[1]
    order = lambda flat: flat[np.argsort(-s.ravel()[flat], kind="stable")]
    chosen: list[int] = []
    if per_node > 0:
        col = np.where(np.isnan(s), -np.inf, s)
        best = np.argsort(-col, axis=0, kind="stable")[:per_node]  # per_node × n chỉ số nguồn
        for r in range(best.shape[0]):
            for b in range(n):
                a = best[r, b]
                if np.isfinite(col[a, b]):
                    chosen.append(a * n + b)
        chosen = order(np.array(chosen, dtype=np.int64)).tolist()[:k]
    taken = set(chosen)
    rest = [f for f in order(np.flatnonzero(~np.isnan(s))).tolist() if f not in taken]
    flat = np.array(chosen + rest[: max(k - len(chosen), 0)], dtype=np.int64)
    rows, cols = np.unravel_index(flat, s.shape)
    return list(zip(rows.tolist(), cols.tolist()))


def statistical_graph(seqs, users, n_nodes: int, n_edges: int, heldout=(), threshold: float = 0.5,
                      min_support: int = 50, per_node: int = 1) -> tuple[sp.csr_matrix, dict]:
    user, node, correct = interactions_of_users(seqs, users)
    att, mas = mastery_matrices(user, node, correct, n_nodes, threshold)
    av, _ = added_value_scores(att, mas, min_support)
    edges = top_k_edges(av, n_edges, exclude_nodes=heldout, per_node=per_node)
    adj = build_adjacency(edges, n_nodes)
    vals = np.array([av[a, b] for a, b in edges])
    info = {"n_users": int(len(np.unique(user))), "n_interactions": int(len(user)),
            "n_candidate_pairs": int(np.isfinite(av).sum()), "n_edges": len(edges),
            "av_min": float(vals.min()) if len(vals) else None,
            "av_max": float(vals.max()) if len(vals) else None}
    return adj, info


# ---------------------------------------------------------------- đồ thị đối chứng âm

def random_graph(n_nodes: int, n_edges: int, seed: int = 0) -> sp.csr_matrix:
    """Đồ thị có hướng ngẫu nhiên đều, đúng n_edges cạnh, không vòng tự thân, không cạnh trùng."""
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    while len(chosen) < n_edges:
        cand = rng.integers(0, n_nodes * n_nodes, size=2 * (n_edges - len(chosen)))
        for c in cand:
            if c // n_nodes != c % n_nodes:
                chosen.add(int(c))
                if len(chosen) == n_edges:
                    break
    flat = np.fromiter(chosen, dtype=np.int64)
    return build_adjacency(list(zip((flat // n_nodes).tolist(), (flat % n_nodes).tolist())), n_nodes)


def dense_graph(n_nodes: int) -> sp.csr_matrix:
    """Đồ thị đầy đủ (mọi cặp khác nhau), như thiết lập dense của GKT gốc."""
    a = np.ones((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(a, 0)
    return sp.csr_matrix(a)


# ---------------------------------------------------------------- so sánh đồ thị

def edge_set(adj: sp.spmatrix) -> set[tuple[int, int]]:
    coo = sp.coo_matrix(adj)
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def compare_graphs(adj: sp.spmatrix, ref: sp.spmatrix) -> dict:
    """Mức trùng cạnh có hướng và vô hướng (bỏ qua chiều) giữa adj và đồ thị tham chiếu."""
    e, r = edge_set(adj), edge_set(ref)
    und = lambda s: {tuple(sorted(x)) for x in s}
    eu, ru = und(e), und(r)
    return {
        "jaccard_directed": len(e & r) / max(len(e | r), 1),
        "jaccard_undirected": len(eu & ru) / max(len(eu | ru), 1),
        "shared_directed": len(e & r),
        "reversed": len({(b, a) for a, b in e} & (r - e)),  # cạnh của adj có chiều ngược với tham chiếu
    }
