"""Tiền xử lý Junyi: ánh xạ chỉ số nút, đồ thị tiên quyết, chuỗi tương tác, phân chia dữ liệu.

Quy ước chiều cạnh: cột `prerequisites` của bài X liệt kê các bài tiên quyết P,
mỗi P sinh một cạnh có hướng P -> X (khái niệm tiên quyết -> khái niệm phụ thuộc).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

LOG_COLUMNS = ["user_id", "exercise", "time_done", "correct"]


# ---------------------------------------------------------------- đồ thị tiên quyết

def load_exercise_table(path: str | Path) -> pd.DataFrame:
    """Đọc bảng bài tập; tên trùng (Junyi 2015: 837 dòng, 835 tên) giữ bản có creation_date mới nhất."""
    table = pd.read_csv(path, encoding="utf-8")
    table = table.sort_values("creation_date", kind="stable")
    table = table.drop_duplicates("name", keep="last")
    return table.sort_index().reset_index(drop=True)


def parse_prerequisites(value) -> list[str]:
    """'a, b' -> ['a', 'b']; ô trống/NaN -> []."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]


def build_node_index(names) -> dict[str, int]:
    """Ánh xạ tên bài tập -> node_id theo thứ tự từ điển, để kết quả tất định."""
    return {name: i for i, name in enumerate(sorted(set(names)))}


@dataclass
class EdgeReport:
    edges: list[tuple[int, int]]
    missing_refs: list[tuple[str, str]] = field(default_factory=list)  # (tiên quyết, bài) không có trong chỉ số
    self_loops: int = 0
    duplicates: int = 0


def build_prerequisite_edges(table: pd.DataFrame, node_index: dict[str, int]) -> EdgeReport:
    seen: set[tuple[int, int]] = set()
    report = EdgeReport(edges=[])
    for name, prereqs in zip(table["name"], table["prerequisites"]):
        if name not in node_index:
            continue
        dst = node_index[name]
        for parent in parse_prerequisites(prereqs):
            if parent not in node_index:
                report.missing_refs.append((parent, name))
                continue
            src = node_index[parent]
            if src == dst:
                report.self_loops += 1
                continue
            if (src, dst) in seen:
                report.duplicates += 1
                continue
            seen.add((src, dst))
            report.edges.append((src, dst))
    return report


def build_adjacency(edges: list[tuple[int, int]], n_nodes: int) -> sp.csr_matrix:
    """Ma trận kề thưa A với A[src, dst] = 1."""
    if edges:
        src, dst = np.array(edges, dtype=np.int64).T
    else:
        src = dst = np.array([], dtype=np.int64)
    data = np.ones(len(src), dtype=np.float32)
    return sp.csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))


def is_dag(adj: sp.csr_matrix) -> bool:
    """Thuật toán Kahn: đồ thị là DAG khi sắp xếp tô-pô đi hết mọi nút."""
    in_deg = np.asarray((adj > 0).sum(axis=0)).ravel().astype(np.int64)
    queue = deque(np.flatnonzero(in_deg == 0).tolist())
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in adj.indices[adj.indptr[u]:adj.indptr[u + 1]]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    return visited == adj.shape[0]


def graph_stats(adj: sp.csr_matrix) -> dict:
    binary = (adj > 0).astype(np.int8)
    in_deg = np.asarray(binary.sum(axis=0)).ravel()
    out_deg = np.asarray(binary.sum(axis=1)).ravel()
    n_comp, labels = connected_components(binary, directed=True, connection="weak")
    comp_sizes = np.bincount(labels)
    _, scc = connected_components(binary, directed=True, connection="strong")
    scc_sizes = np.bincount(scc)
    cycles = [np.flatnonzero(scc == c).tolist() for c in np.flatnonzero(scc_sizes > 1)]
    return {
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(binary.nnz),
        "density": float(binary.nnz / max(adj.shape[0] * (adj.shape[0] - 1), 1)),
        "n_isolated": int(((in_deg + out_deg) == 0).sum()),
        "n_roots": int(((in_deg == 0) & (out_deg > 0)).sum()),  # khái niệm nền tảng: không có tiên quyết
        "n_leaves": int(((out_deg == 0) & (in_deg > 0)).sum()),
        "in_degree_max": int(in_deg.max(initial=0)),
        "out_degree_max": int(out_deg.max(initial=0)),
        "in_degree_mean": float(in_deg.mean()) if len(in_deg) else 0.0,
        "n_weak_components": int(n_comp),
        "largest_component": int(comp_sizes.max(initial=0)),
        "is_dag": is_dag(binary.tocsr()),
        # thành phần liên thông mạnh có hơn một nút = các nút nằm trên chu trình
        "cycles": cycles,
    }


# ---------------------------------------------------------------- nhật ký tương tác

def load_problem_log(path: str | Path, node_index: dict[str, int], chunksize: int = 2_000_000) -> tuple[pd.DataFrame, dict]:
    """Đọc nhật ký theo khối, giữ 4 cột, bỏ tương tác thuộc bài không có trong chỉ số nút."""
    parts, n_rows, n_unknown = [], 0, 0
    unknown_names: set[str] = set()
    reader = pd.read_csv(
        path,
        usecols=LOG_COLUMNS,
        dtype={"user_id": np.int64, "exercise": str, "time_done": np.int64, "correct": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        n_rows += len(chunk)
        node = chunk["exercise"].map(node_index)
        unknown = node.isna()
        n_unknown += int(unknown.sum())
        unknown_names.update(chunk.loc[unknown, "exercise"].unique().tolist())
        chunk = chunk.loc[~unknown]
        parts.append(pd.DataFrame({
            "user_id": chunk["user_id"].to_numpy(np.int64),
            "node_id": node[~unknown].to_numpy(np.int32),
            "time_done": chunk["time_done"].to_numpy(np.int64),
            "correct": chunk["correct"].str.lower().eq("true").to_numpy(np.int8),
        }))
    log = pd.concat(parts, ignore_index=True)
    info = {"n_rows_raw": n_rows, "n_rows_unknown_exercise": n_unknown, "unknown_exercises": sorted(unknown_names)}
    return log, info


def sort_and_filter(log: pd.DataFrame, min_interactions: int = 3) -> pd.DataFrame:
    """Sắp theo (người học, thời điểm) giữ thứ tự gốc khi trùng thời điểm; bỏ người học quá ít tương tác."""
    counts = log["user_id"].map(log["user_id"].value_counts())
    log = log.loc[counts >= min_interactions]
    order = np.lexsort((np.arange(len(log)), log["time_done"].to_numpy(), log["user_id"].to_numpy()))
    return log.iloc[order].reset_index(drop=True)


def build_sequences(log: pd.DataFrame, max_len: int = 200, min_len: int = 3) -> dict[str, np.ndarray]:
    """Chia chuỗi của mỗi người học thành các đoạn liên tiếp dài tối đa max_len.

    Đoạn cuối ngắn hơn min_len bị bỏ. Kết quả lưu dạng nối liền + mảng offsets:
    đoạn i là questions[offsets[i]:offsets[i+1]].
    """
    users = log["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    seg_starts, seg_ends = [], []
    for s, e in zip(starts, ends):
        for a in range(s, e, max_len):
            b = min(a + max_len, e)
            if b - a >= min_len:
                seg_starts.append(a)
                seg_ends.append(b)
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    seg_ends = np.asarray(seg_ends, dtype=np.int64)
    keep = np.zeros(len(users), dtype=bool)
    for a, b in zip(seg_starts, seg_ends):
        keep[a:b] = True
    lengths = seg_ends - seg_starts
    return {
        "questions": log["node_id"].to_numpy(np.int16)[keep],
        "responses": log["correct"].to_numpy(np.int8)[keep],
        "offsets": np.r_[0, np.cumsum(lengths)].astype(np.int64),
        "seq_user": users[seg_starts],
    }


def split_users(user_ids, n_folds: int = 5, test_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Phân chia theo người học: test_frac làm tập kiểm tra, phần còn lại chia n_folds phần.

    Ở phần k: tập kiểm định = phần k, tập huấn luyện = các phần còn lại.
    Cột `split` nhận 'test' hoặc chỉ số phần '0'..'n_folds-1'.
    """
    users = np.unique(np.asarray(user_ids))
    rng = np.random.default_rng(seed)
    users = users[rng.permutation(len(users))]
    n_test = int(round(len(users) * test_frac))
    split = np.empty(len(users), dtype=object)
    split[:n_test] = "test"
    split[n_test:] = (np.arange(len(users) - n_test) % n_folds).astype(str)
    return pd.DataFrame({"user_id": users, "split": split}).sort_values("user_id", ignore_index=True)


# ---------------------------------------------------------------- lưu kết quả

def save_processed(out_dir: str | Path, *, nodes: pd.DataFrame, edges: list[tuple[int, int]],
                   adj: sp.csr_matrix, sequences: dict[str, np.ndarray], splits: pd.DataFrame,
                   stats: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / "nodes.csv", index=False)
    pd.DataFrame(edges, columns=["src", "dst"]).to_csv(out / "edges_expert.csv", index=False)
    sp.save_npz(out / "adj_expert.npz", adj)
    np.savez_compressed(out / "sequences.npz", **sequences)
    splits.to_csv(out / "user_split.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
