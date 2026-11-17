import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from graphk.analysis import _midrank_fast, auc_by_group, delong_paired, node_groups, pairwise_tests, results_table
from graphk.preprocess import build_adjacency


def test_midrank_ties():
    assert _midrank_fast(np.array([0.3, 0.1, 0.3, 0.2])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    good = y + rng.normal(0, 0.8, 4000)
    bad = y + rng.normal(0, 3.0, 4000)
    r = delong_paired(y, good, bad)
    assert r["auc_1"] == pytest.approx(roc_auc_score(y, good))
    assert r["auc_2"] == pytest.approx(roc_auc_score(y, bad))
    assert r["delta"] > 0 and r["p_value"] < 1e-6
    same = delong_paired(y, good, good)
    assert same["delta"] == 0 and same["p_value"] == 1.0


def test_pairwise_tests_mask_and_order_check():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    q = rng.integers(0, 4, 500)
    preds = {"a": {"y": y, "q": q, "p": y + rng.normal(0, 1, 500)},
             "b": {"y": y, "q": q, "p": rng.random(500)}}
    t = pairwise_tests(preds, [("a", "b")], mask_fn=lambda qq: qq == 0)
    assert t.loc[0, "số dự đoán"] == int((q == 0).sum())
    preds["c"] = {"y": y[::-1], "q": q, "p": rng.random(500)}
    with pytest.raises(ValueError):
        pairwise_tests(preds, [("a", "c")])


def test_node_groups_and_auc_by_group():
    g = node_groups(build_adjacency([(0, 1), (0, 2), (0, 3)], 5), build_adjacency([(1, 2)], 5))
    assert g["nhóm chuyên gia"].tolist() == ["bậc cao (≥3)", "bậc thấp (1–2)", "bậc thấp (1–2)", "bậc thấp (1–2)", "cô lập"]
    assert g["cô lập trong đồ thị thống kê"].tolist() == [True, False, False, True, True]
    rng = np.random.default_rng(2)
    y, q = rng.integers(0, 2, 300), rng.integers(0, 5, 300)
    tab = auc_by_group({"m": {"y": y, "q": q, "p": rng.random(300)}}, g["nhóm chuyên gia"].to_numpy())
    assert set(tab.index) == {"bậc cao (≥3)", "bậc thấp (1–2)", "cô lập"}


def test_results_table(tmp_path):
    res = {"name": "x", "epochs": 2, "best_val_auc": 0.7, "history": [{"train_seconds": 60}, {"train_seconds": 120}],
           "test": {k: {"auc": 0.7, "acc": 0.8, "rmse": 0.4, "n": 10} for k in ("all", "heldout", "non_heldout")}}
    from graphk.analysis import load_results
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "results.json").write_text(json.dumps(res))
    t = results_table(load_results(tmp_path))
    assert t.loc[0, "phút huấn luyện"] == pytest.approx(3.0)


def test_select_case_study_prefers_prerequisite_chains():
    from graphk.analysis import select_case_study
    adj = build_adjacency([(0, 1)], 3)
    # đoạn 0: 1 xuất hiện trước 0 (không tính); đoạn 1: 0 rồi 1 với kết quả 1 lẫn lộn
    q = [1, 0, 1, 2, 0, 1, 1, 2]
    r = [1, 1, 1, 1, 1, 0, 1, 0]
    seqs = {"questions": np.array(q), "responses": np.array(r), "offsets": np.array([0, 4, 8]), "seq_user": np.array([0, 1])}
    seg, edges = select_case_study(seqs, np.array([0, 1]), adj, min_len=3, max_len=10)
    assert seg == 1 and edges == [(0, 1)]


def test_pairwise_tests_empty_has_columns():
    t = pairwise_tests({}, [("a", "b")])
    assert len(t) == 0 and "p_value" in t.columns
