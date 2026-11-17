import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from graphk.models import build_model  # noqa: E402
from graphk.models.gkt import BWD, FWD, add_virtual_root, neighbor_lists  # noqa: E402
from graphk.preprocess import build_adjacency  # noqa: E402

N = 6
ADJ = build_adjacency([(0, 1), (1, 2), (0, 3)], N)  # 4, 5 cô lập


def test_neighbor_lists_directions_and_norm():
    ptr, nbr, typ, w = neighbor_lists(ADJ, bidirectional=True, norm="row")
    sl = lambda i: slice(int(ptr[i]), int(ptr[i + 1]))
    # nút 0: phụ thuộc 1, 3 (FWD); không có tiên quyết
    assert sorted(nbr[sl(0)].tolist()) == [1, 3] and set(typ[sl(0)].tolist()) == {FWD}
    assert w[sl(0)].tolist() == [0.5, 0.5]
    # nút 1: phụ thuộc 2 (FWD), tiên quyết 0 (BWD)
    assert nbr[sl(1)].tolist() == [2, 0] and typ[sl(1)].tolist() == [FWD, BWD]
    # nút cô lập không có lân cận
    assert int(ptr[5] - ptr[4]) == 0
    _, _, typ_uni, _ = neighbor_lists(ADJ, bidirectional=False)
    assert (typ_uni == BWD).sum() == 0


def test_virtual_root():
    a = add_virtual_root(ADJ)
    assert a.shape == (N + 1, N + 1)
    # gốc ảo nối tới các nút có bậc vào 0: 0, 4, 5
    assert sorted(a[N].indices.tolist()) == [0, 4, 5]


@pytest.mark.parametrize("name,kwargs", [
    ("gkt", {"hidden": 8, "emb": 8}),
    ("gkt", {"hidden": 8, "emb": 8, "isolated": "root", "bidirectional": False, "norm": "none"}),
    ("dkt", {"emb": 8, "hidden": 8}),
    ("akt", {"d_model": 16, "n_blocks": 1, "n_heads": 2, "d_ff": 16, "final_fc_dim": 16}),
])
def test_no_future_leakage(name, kwargs):
    """logit[:, t] dự đoán r[:, t+1]: không được đổi khi sửa r[:, t+1:], phải đổi được theo q[:, t+1]."""
    torch.manual_seed(0)
    model = build_model(name, N, ADJ, dropout=0.0, **kwargs).eval()
    q = torch.randint(0, N, (3, 10))
    r = torch.randint(0, 2, (3, 10))
    out = model(q, r)
    assert out.shape == (3, 9)
    t = 4
    r2 = r.clone()
    r2[:, t + 1:] = 1 - r2[:, t + 1:]
    out2 = model(q, r2)
    assert torch.allclose(out[:, :t + 1], out2[:, :t + 1], atol=1e-5)
    assert not torch.allclose(out[:, t + 1:], out2[:, t + 1:], atol=1e-5)


def test_gkt_isolated_answer_does_not_touch_other_nodes():
    """Trả lời nút cô lập 4 không được thay đổi dự đoán cho các nút khác ở bước sau."""
    torch.manual_seed(0)
    model = build_model("gkt", N, ADJ, hidden=8, emb=8, dropout=0.0).eval()
    q = torch.tensor([[4, 2], [4, 4]])
    out0 = model(q, torch.tensor([[0, 0], [0, 0]]))
    out1 = model(q, torch.tensor([[1, 0], [1, 0]]))
    assert torch.allclose(out0[0], out1[0])          # dự đoán nút 2 không đổi
    assert not torch.allclose(out0[1], out1[1])      # dự đoán chính nút 4 thay đổi


def _toy_processed(tmp_path):
    rng = np.random.default_rng(0)
    q, r, off, su = [], [], [0], []
    for u in range(40):
        L = int(rng.integers(5, 15))
        q += rng.integers(0, N, L).tolist()
        r += rng.integers(0, 2, L).tolist()
        off.append(off[-1] + L)
        su.append(u)
    proc = tmp_path / "proc"
    (proc / "graphs").mkdir(parents=True)
    np.savez(proc / "sequences.npz", questions=np.array(q, np.int16), responses=np.array(r, np.int8),
             offsets=np.array(off), seq_user=np.array(su))
    pd.DataFrame({"node_id": range(N), "exercise_name": list("abcdef"), "n_interactions": 10}).to_csv(proc / "nodes.csv", index=False)
    import scipy.sparse as sp
    sp.save_npz(proc / "adj_expert.npz", ADJ)
    split = ["test" if u < 8 else str(u % 5) for u in range(40)]
    pd.DataFrame({"user_id": range(40), "split": split}).to_csv(proc / "user_split.csv", index=False)
    pd.DataFrame({"node_id": [5]}).to_csv(proc / "graphs" / "heldout_concepts.csv", index=False)
    sp.save_npz(proc / "graphs" / "adj_stat_fold0.npz", build_adjacency([(2, 3)], N))
    return proc


@pytest.mark.parametrize("model,graph", [("gkt", "expert"), ("gkt", "stat"), ("dkt", "none")])
def test_run_end_to_end_and_resume(tmp_path, model, graph):
    from graphk.train import Config, run
    proc = _toy_processed(tmp_path)
    cfg = Config(proc_dir=str(proc), runs_dir=str(tmp_path / "runs"), model=model, graph=graph,
                 batch_size=8, max_epochs=2, patience=5, model_kwargs={"hidden": 8, "emb": 8})
    res = run(cfg, log=lambda *_: None)
    assert res["epochs"] == 2 and res["test"]["all"]["n"] > 0
    saved = json.loads((tmp_path / "runs" / cfg.name / "results.json").read_text())
    assert saved["name"] == cfg.name
    # chạy lại với max_epochs lớn hơn: học tiếp từ epoch 2, không bắt đầu lại
    cfg.max_epochs = 3
    res2 = run(cfg, log=lambda *_: None)
    assert res2["epochs"] == 3 and len(res2["history"]) == 3


@pytest.mark.parametrize("name,kwargs", [("gkt", {"hidden": 8, "emb": 8}), ("dkt", {"emb": 8, "hidden": 8})])
def test_trace_matches_forward(name, kwargs):
    """trace(q, r)[t, q[t+1]] phải bằng sigmoid(forward(q, r))[t]."""
    torch.manual_seed(0)
    model = build_model(name, N, ADJ, dropout=0.0, **kwargs).eval()
    q = torch.randint(0, N, (12,))
    r = torch.randint(0, 2, (12,))
    tr = model.trace(q, r)
    assert tr.shape == (12, N)
    fwd = torch.sigmoid(model(q.unsqueeze(0), r.unsqueeze(0)))[0]
    assert torch.allclose(tr[torch.arange(11), q[1:]], fwd, atol=1e-5)
