"""Huấn luyện và đánh giá mô hình truy vết tri thức trên dữ liệu đã tiền xử lý.

Giao thức (dàn ý mục 4.1): ở phần k, huấn luyện trên các phần khác k (trừ tập kiểm tra), dừng sớm theo
AUC trên tập kiểm định = phần k, rồi đánh giá mô hình tốt nhất trên tập kiểm tra chung.
Điểm kiểm tra được lưu sau mỗi epoch; chạy lại cùng cấu hình sẽ học tiếp từ điểm kiểm tra gần nhất.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from graphk.graphs import dense_graph, load_processed
from graphk.models import build_model


@dataclass
class Config:
    proc_dir: str
    runs_dir: str
    model: str = "gkt"              # gkt | dkt | akt
    graph: str = "expert"           # expert | stat | random | dense | none
    fold: int = 0
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 20
    patience: int = 3
    max_train_batches: int | None = None   # giới hạn số lô mỗi epoch (chạy thử nhanh)
    max_eval_batches: int | None = None
    model_kwargs: dict = field(default_factory=dict)
    tag: str = ""

    @property
    def name(self) -> str:
        g = self.graph if self.model == "gkt" else "none"
        extra = f"-{self.tag}" if self.tag else ""
        return f"{self.model}-{g}-fold{self.fold}-seed{self.seed}{extra}"


# ---------------------------------------------------------------- dữ liệu

def segment_indices(seqs: dict, users) -> np.ndarray:
    return np.flatnonzero(np.isin(seqs["seq_user"], np.asarray(users)))


def make_batches(seqs: dict, seg_idx: np.ndarray, batch_size: int, shuffle: bool, rng=None) -> list[np.ndarray]:
    """Gom các đoạn có độ dài gần nhau vào cùng lô (giảm số bước đệm); xáo thứ tự lô khi huấn luyện."""
    lengths = np.diff(seqs["offsets"])[seg_idx]
    noise = rng.random(len(seg_idx)) if shuffle else np.zeros(len(seg_idx))
    order = seg_idx[np.lexsort((noise, lengths))]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(seqs: dict, batch: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Đệm phải về độ dài lớn nhất của lô; mask đánh dấu vị trí đích hợp lệ (B, T-1)."""
    off = seqs["offsets"]
    lens = off[batch + 1] - off[batch]
    t = int(lens.max())
    q = np.zeros((len(batch), t), dtype=np.int64)
    r = np.zeros((len(batch), t), dtype=np.int64)
    for i, s in enumerate(batch):
        q[i, :lens[i]] = seqs["questions"][off[s]:off[s + 1]]
        r[i, :lens[i]] = seqs["responses"][off[s]:off[s + 1]]
    mask = np.arange(1, t)[None, :] < lens[:, None]
    return (torch.from_numpy(q).to(device), torch.from_numpy(r).to(device),
            torch.from_numpy(mask).to(device))


def load_graph(cfg: Config, proc: Path, adj_expert: sp.csr_matrix) -> sp.csr_matrix | None:
    if cfg.model != "gkt":
        return None
    n = adj_expert.shape[0]
    if cfg.graph == "expert":
        return adj_expert
    if cfg.graph in ("stat", "random"):
        return sp.load_npz(proc / "graphs" / f"adj_{cfg.graph}_fold{cfg.fold}.npz").tocsr()
    if cfg.graph == "dense":
        return dense_graph(n)
    raise ValueError(f"Đồ thị không hỗ trợ: {cfg.graph}")


# ---------------------------------------------------------------- đánh giá

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": None, "acc": None, "rmse": None}
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)),
            "acc": float(((p >= 0.5) == y).mean()), "rmse": float(np.sqrt(((p - y) ** 2).mean()))}


@torch.no_grad()
def predict(model: nn.Module, seqs: dict, batches, device, max_batches=None):
    model.eval()
    ys, ps, qs = [], [], []
    for b_i, batch in enumerate(batches):
        if max_batches is not None and b_i >= max_batches:
            break
        q, r, mask = collate(seqs, batch, device)
        p = torch.sigmoid(model(q, r))
        ys.append(r[:, 1:][mask].cpu().numpy())
        ps.append(p[mask].float().cpu().numpy())
        qs.append(q[:, 1:][mask].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(qs)


def evaluate(y, p, q, heldout: np.ndarray) -> dict:
    h = np.isin(q, heldout)
    return {"all": metrics(y, p), "heldout": metrics(y[h], p[h]), "non_heldout": metrics(y[~h], p[~h])}


# ---------------------------------------------------------------- huấn luyện

def run(cfg: Config, data: dict | None = None, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = Path(cfg.proc_dir)
    run_dir = Path(cfg.runs_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    data = data or load_processed(proc)
    seqs, splits = data["sequences"], data["splits"]
    n_nodes = len(data["nodes"])
    heldout = np.loadtxt(proc / "graphs" / "heldout_concepts.csv", delimiter=",", skiprows=1, usecols=0, dtype=np.int64)

    train_users = splits.loc[(splits.split != "test") & (splits.split != str(cfg.fold)), "user_id"]
    val_users = splits.loc[splits.split == str(cfg.fold), "user_id"]
    test_users = splits.loc[splits.split == "test", "user_id"]
    train_idx = segment_indices(seqs, train_users)
    val_batches = make_batches(seqs, segment_indices(seqs, val_users), cfg.batch_size, shuffle=False)
    test_batches = make_batches(seqs, segment_indices(seqs, test_users), cfg.batch_size, shuffle=False)

    adj = load_graph(cfg, proc, data["adj_expert"])
    model = build_model(cfg.model, n_nodes, adj, **cfg.model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    log(f"[{cfg.name}] thiết bị={device} | tham số={sum(p.numel() for p in model.parameters()):,} | "
        f"đoạn huấn luyện={len(train_idx):,} | lô kiểm định={len(val_batches)} | lô kiểm tra={len(test_batches)}")

    state = {"epoch": 0, "best_auc": -1.0, "bad_epochs": 0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        rng = np.random.default_rng(cfg.seed + state["epoch"])
        log(f"Học tiếp từ epoch {state['epoch']} (AUC kiểm định tốt nhất {state['best_auc']:.4f})")

    while state["epoch"] < cfg.max_epochs and state["bad_epochs"] < cfg.patience:
        model.train()
        batches = make_batches(seqs, train_idx, cfg.batch_size, shuffle=True, rng=rng)
        if cfg.max_train_batches is not None:
            batches = batches[:cfg.max_train_batches]
        t0, total, count = time.time(), 0.0, 0
        for b_i, batch in enumerate(batches):
            q, r, mask = collate(seqs, batch, device)
            logit = model(q, r)
            loss = loss_fn(logit[mask], r[:, 1:][mask].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * int(mask.sum())
            count += int(mask.sum())
            if b_i in (9, 99) or (b_i + 1) % 500 == 0:
                sec = (time.time() - t0) / (b_i + 1)
                log(f"  lô {b_i + 1}/{len(batches)} | mất mát {total / count:.4f} | {sec:.3f} s/lô | "
                    f"ước tính epoch {sec * len(batches) / 60:.1f} phút")
        train_time = time.time() - t0
        y, p, qq = predict(model, seqs, val_batches, device, cfg.max_eval_batches)
        val = evaluate(y, p, qq, heldout)
        state["epoch"] += 1
        auc = val["all"]["auc"] or 0.0
        improved = auc > state["best_auc"]
        if improved:
            state["best_auc"], state["bad_epochs"] = auc, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": total / max(count, 1),
                                 "train_seconds": train_time, "val": val})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, last)
        log(f"Epoch {state['epoch']}: mất mát {total / max(count, 1):.4f} | AUC kiểm định {auc:.4f} | "
            f"AUC khái niệm giữ lại {val['heldout']['auc']} | {train_time / 60:.1f} phút{' *' if improved else ''}")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    y, p, qq = predict(model, seqs, test_batches, device, cfg.max_eval_batches)
    test = evaluate(y, p, qq, heldout)
    np.savez_compressed(run_dir / "test_predictions.npz", y=y.astype(np.int8), p=p.astype(np.float32), q=qq.astype(np.int16))
    result = {"name": cfg.name, "config": asdict(cfg), "best_val_auc": state["best_auc"],
              "epochs": state["epoch"], "history": state["history"], "test": test}
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"[{cfg.name}] AUC kiểm tra {test['all']['auc']:.4f} | khái niệm giữ lại {test['heldout']['auc']} | "
        f"ACC {test['all']['acc']:.4f} | RMSE {test['all']['rmse']:.4f}")
    return result
