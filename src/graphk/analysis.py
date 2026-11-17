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
