"""Phân tích kết quả thực nghiệm: bảng tổng hợp, AUC theo nhóm nút, kiểm định DeLong ghép cặp.

Dự đoán trên tập kiểm tra (test_predictions.npz) của mọi mô hình có cùng thứ tự: các đoạn được sắp
theo độ dài một cách tất định (train.make_batches với shuffle=False) và trải phẳng theo hàng, không phụ
thuộc cỡ lô. Nhờ vậy có thể so sánh ghép cặp từng dự đoán giữa hai mô hình.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- đọc kết quả

def load_results(runs_dir: str | Path) -> dict[str, dict]:
    """{tên thực nghiệm: nội dung results.json} cho mọi thực nghiệm đã chạy xong."""
    out = {}
    for f in sorted(Path(runs_dir).glob("*/results.json")):
        res = json.loads(f.read_text())
        out[res["name"]] = res
    return out


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        t = res["test"]
        rows.append({
            "thực nghiệm": name,
            "epoch": res["epochs"],
            "AUC kiểm định": res["best_val_auc"],
            "AUC kiểm tra": t["all"]["auc"],
            "ACC kiểm tra": t["all"]["acc"],
            "RMSE kiểm tra": t["all"]["rmse"],
            "AUC khái niệm giữ lại": t["heldout"]["auc"],
            "AUC khái niệm còn lại": t["non_heldout"]["auc"],
            "phút huấn luyện": sum(h["train_seconds"] for h in res["history"]) / 60,
        })
    return pd.DataFrame(rows)


def load_predictions(runs_dir: str | Path, name: str) -> dict[str, np.ndarray]:
    return dict(np.load(Path(runs_dir) / name / "test_predictions.npz"))


# ---------------------------------------------------------------- nhóm nút

def node_groups(adj_expert: sp.spmatrix, adj_stat: sp.spmatrix | None = None) -> pd.DataFrame:
    """Nhóm khái niệm theo bậc (vào + ra) trong đồ thị chuyên gia; tùy chọn: cô lập trong đồ thị thống kê."""
    a = sp.csr_matrix(adj_expert) > 0
    deg = np.asarray(a.sum(0)).ravel() + np.asarray(a.sum(1)).ravel()
    group = np.where(deg == 0, "cô lập", np.where(deg <= 2, "bậc thấp (1–2)", "bậc cao (≥3)"))
    df = pd.DataFrame({"node_id": np.arange(a.shape[0]), "bậc chuyên gia": deg, "nhóm chuyên gia": group})
    if adj_stat is not None:
        s = sp.csr_matrix(adj_stat) > 0
        sdeg = np.asarray(s.sum(0)).ravel() + np.asarray(s.sum(1)).ravel()
        df["cô lập trong đồ thị thống kê"] = sdeg == 0
    return df


def auc_by_group(preds: dict[str, dict[str, np.ndarray]], node_group: np.ndarray) -> pd.DataFrame:
    """AUC của từng mô hình trên từng nhóm khái niệm đích (node_group[q] là nhãn nhóm của khái niệm q)."""
    rows = []
    for name, d in preds.items():
        g = node_group[d["q"].astype(np.int64)]
        for label in pd.unique(node_group):
            m = g == label
            if m.sum() and len(np.unique(d["y"][m])) == 2:
                rows.append({"thực nghiệm": name, "nhóm": label, "số dự đoán": int(m.sum()),
                             "AUC": roc_auc_score(d["y"][m], d["p"][m])})
    return pd.DataFrame(rows).pivot(index="nhóm", columns="thực nghiệm", values="AUC")


# ---------------------------------------------------------------- kiểm định DeLong ghép cặp

def _midrank_fast(x: np.ndarray) -> np.ndarray:
    """Hạng trung bình (ties nhận hạng trung bình), cài đặt vector hóa cho mảng lớn."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    boundaries = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])
    starts, ends = boundaries[:-1], boundaries[1:]
    avg = 0.5 * (starts + ends - 1) + 1
    ranks_sorted = np.repeat(avg, ends - starts)
    out = np.empty(len(x), dtype=np.float64)
    out[order] = ranks_sorted
    return out


def delong_paired(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Kiểm định DeLong cho hiệu AUC của hai mô hình trên cùng tập dự đoán (Sun & Xu, 2014).

    Lưu ý: giả định các dự đoán độc lập, bỏ qua tương quan giữa các tương tác của cùng một người học,
    nên giá trị p có xu hướng nhỏ hơn thực tế; cần đọc kèm độ lớn hiệu AUC.
    """
    y = np.asarray(y).astype(bool)
    pos, neg = y, ~y
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p, dtype=np.float64)
        tx, ty, tz = _midrank_fast(p[pos]), _midrank_fast(p[neg]), _midrank_fast(p)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((tz[pos] - tx) / n)
        v01.append(1.0 - (tz[neg] - ty) / m)
    s10 = np.cov(np.vstack(v10))
    s01 = np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else (0.0 if diff == 0 else np.inf)
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "delta": float(diff),
            "se": float(np.sqrt(max(var, 0.0))), "z": float(z), "p_value": float(2 * norm.sf(abs(z)))}


def pairwise_tests(preds: dict[str, dict[str, np.ndarray]], pairs: list[tuple[str, str]],
                   mask_fn=None) -> pd.DataFrame:
    """Kiểm định DeLong cho từng cặp (mô hình 1, mô hình 2); mask_fn(q) chọn tập con dự đoán."""
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        da, db = preds[a], preds[b]
        if not np.array_equal(da["y"], db["y"]) or not np.array_equal(da["q"], db["q"]):
            raise ValueError(f"Dự đoán của {a} và {b} không cùng thứ tự")
        m = mask_fn(da["q"]) if mask_fn else np.ones(len(da["y"]), dtype=bool)
        r = delong_paired(da["y"][m], da["p"][m], db["p"][m])
        rows.append({"mô hình 1": a, "mô hình 2": b, "số dự đoán": int(m.sum()), **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- case study

def select_case_study(seqs: dict, segments: np.ndarray, adj_expert: sp.spmatrix, min_len: int = 25,
                      max_len: int = 60, max_scan: int = 20000) -> tuple[int, list[tuple[int, int]]]:
    """Chọn tất định một đoạn trong tập kiểm tra minh họa tốt việc lan truyền theo cạnh tiên quyết.

    Điểm của đoạn = số cạnh chuyên gia (u -> v) mà u xuất hiện trước v trong đoạn và kết quả trên v
    có cả đúng lẫn sai. Lấy đoạn điểm cao nhất; hòa thì lấy đoạn ngắn hơn, rồi chỉ số nhỏ hơn.
    """
    a = sp.csr_matrix(adj_expert)
    off = seqs["offsets"]
    candidates = []
    for s in np.sort(segments)[:max_scan]:
        n = int(off[s + 1] - off[s])
        if not (min_len <= n <= max_len):
            continue
        q = seqs["questions"][off[s]:off[s + 1]].astype(np.int64)
        r = seqs["responses"][off[s]:off[s + 1]]
        first = {}
        for t, c in enumerate(q):
            first.setdefault(int(c), t)
        edges = [(u, int(v)) for u, tu in first.items() for v in a.indices[a.indptr[u]:a.indptr[u + 1]]
                 if int(v) in first and first[int(v)] > tu and len(np.unique(r[q == v])) == 2]
        if edges:
            candidates.append((len(edges), -n, -int(s), edges))
    if not candidates:
        raise ValueError("Không tìm được đoạn phù hợp; nới min_len/max_len hoặc max_scan")
    best = max(candidates, key=lambda c: c[:3])
    return -best[2], best[3]


def load_trained(runs_dir: str | Path, name: str, proc_dir: str | Path, data: dict):
    """Tải mô hình tốt nhất (best.pt) của một thực nghiệm, ở chế độ đánh giá trên CPU."""
    import torch
    from graphk.models import build_model
    from graphk.train import Config, load_graph
    cfg_dict = json.loads((Path(runs_dir) / name / "config.json").read_text())
    cfg = Config(**cfg_dict)
    adj = load_graph(cfg, Path(proc_dir), data["adj_expert"])
    model = build_model(cfg.model, len(data["nodes"]), adj, **cfg.model_kwargs)
    model.load_state_dict(torch.load(Path(runs_dir) / name / "best.pt", map_location="cpu"))
    return model.eval()


def case_study_traces(models: dict, seqs: dict, segment: int, focus: list[int]) -> dict:
    """Mức thành thạo dự đoán sau từng bước cho các khái niệm focus, theo từng mô hình."""
    import torch
    off = seqs["offsets"]
    q = torch.from_numpy(seqs["questions"][off[segment]:off[segment + 1]].astype(np.int64))
    r = torch.from_numpy(seqs["responses"][off[segment]:off[segment + 1]].astype(np.int64))
    out = {"q": q.tolist(), "r": r.tolist(), "focus": list(map(int, focus)), "traces": {}, "next_pred": {}}
    for label, m in models.items():
        tr = m.trace(q, r).numpy()                            # (T, n_nodes): sau bước t
        out["traces"][label] = tr[:, focus].T.round(4).tolist()  # (len(focus), T)
        out["next_pred"][label] = tr[np.arange(len(q) - 1), q[1:].numpy()].round(4).tolist()
    return out


def plot_case_study(case: dict, names: list[str], path: str | Path | None = None):
    """Bản đồ nhiệt mức thành thạo dự đoán (khái niệm focus × bước) cho từng mô hình."""
    import matplotlib.pyplot as plt
    labels = list(case["traces"])
    T = len(case["q"])
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 1.2 + 1.1 * len(labels) * max(1, len(names)) / 3),
                             sharex=True, squeeze=False)
    for ax, lab in zip(axes[:, 0], labels):
        im = ax.imshow(np.array(case["traces"][lab]), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for t, (c, a) in enumerate(zip(case["q"], case["r"])):
            if c in case["focus"]:
                ax.scatter(t, case["focus"].index(c), marker="o" if a else "x",
                           color="white" if a else "red", s=28, linewidths=1.5)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(lab, fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("bước (o: trả lời đúng, x: trả lời sai trên khái niệm đang xét)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="P(trả lời đúng)", shrink=0.8)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_graph_comparison(adj: dict, av_groups: dict, path: str | Path | None = None):
    """Hình 2: phân phối bậc (vào + ra) của các đồ thị và phân phối giá trị gia tăng theo nhóm cạnh."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, m in adj.items():
        b = sp.csr_matrix(m) > 0
        deg = np.asarray(b.sum(0)).ravel() + np.asarray(b.sum(1)).ravel()
        a1.hist(np.clip(deg, 0, 20), bins=np.arange(22) - 0.5, histtype="step", linewidth=1.8,
                label=f"{label} (cô lập: {(deg == 0).sum()})")
    a1.set_xlabel("bậc vào + bậc ra (cắt tại 20)"); a1.set_ylabel("số khái niệm"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title("Phân phối bậc")
    data = [np.asarray(v)[np.isfinite(v)] for v in av_groups.values()]
    a2.boxplot(data, showfliers=False)
    a2.set_xticks(range(1, len(data) + 1), list(av_groups), fontsize=9)
    a2.axhline(0, color="gray", linewidth=0.8); a2.set_ylabel("giá trị gia tăng AV"); a2.grid(alpha=0.3)
    a2.set_title("Giá trị gia tăng theo nhóm cặp khái niệm")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
