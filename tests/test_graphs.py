import numpy as np
import pandas as pd
import pytest

from graphk.graphs import (
    added_value_scores,
    choose_heldout_concepts,
    compare_graphs,
    dense_graph,
    fold_users,
    interactions_of_users,
    mastery_matrices,
    random_graph,
    statistical_graph,
    top_k_edges,
)
from graphk.preprocess import build_adjacency


def _seqs(rows):
    """rows: danh sách (user, [(node, correct), ...]) -> định dạng sequences.npz, mỗi người học một đoạn."""
    q, r, off, su = [], [], [0], []
    for u, items in rows:
        q += [n for n, _ in items]
        r += [c for _, c in items]
        off.append(off[-1] + len(items))
        su.append(u)
    return {"questions": np.array(q, np.int16), "responses": np.array(r, np.int8),
            "offsets": np.array(off, np.int64), "seq_user": np.array(su, np.int64)}


def test_fold_users_excludes_validation_and_test():
    splits = pd.DataFrame({"user_id": [1, 2, 3, 4], "split": ["0", "1", "test", "1"]})
    assert sorted(fold_users(splits, 1)) == [1]
    assert sorted(fold_users(splits, None)) == [1, 2, 4]


def test_interactions_of_users():
    seqs = _seqs([(10, [(0, 1), (1, 0)]), (20, [(2, 1)])])
    u, n, c = interactions_of_users(seqs, [20])
    assert u.tolist() == [20] and n.tolist() == [2] and c.tolist() == [1]


def test_mastery_and_added_value():
    # Người học thành thạo 0 thì thành thạo 1; người học không thành thạo 0 thì không thành thạo 1.
    rows = [(u, [(0, 1), (1, 1)]) for u in range(6)] + [(u, [(0, 0), (1, 0)]) for u in range(6, 12)]
    user, node, correct = interactions_of_users(_seqs(rows), range(12))
    att, mas = mastery_matrices(user, node, correct, n_nodes=3)
    assert att.shape == (12, 3) and att.sum() == 24 and mas.sum() == 12
    av, support = added_value_scores(att, mas, min_support=1)
    # P(1 | 0) = 1, P(1) = 0.5 -> AV = 0.5
    assert av[0, 1] == pytest.approx(0.5)
    assert support[0, 1] == 6
    assert np.isnan(av[0, 0]) and np.isnan(av[0, 2])  # đường chéo; khái niệm 2 không ai làm
    av_strict, _ = added_value_scores(att, mas, min_support=7)
    assert np.isnan(av_strict[0, 1])


def test_top_k_edges_excludes_heldout_and_nonpositive():
    s = np.array([[np.nan, 0.9, 0.5], [0.1, np.nan, -0.2], [0.8, 0.0, np.nan]])
    assert top_k_edges(s, 2, per_node=0) == [(0, 1), (2, 0)]
    assert top_k_edges(s, 10, exclude_nodes=[2], per_node=0) == [(0, 1), (1, 0)]


def test_top_k_edges_per_node_coverage():
    # Toàn cục thì 3 cạnh tốt nhất đều vào nút 1; per_node=1 bảo đảm nút 2 và 3 cũng có cạnh vào.
    s = np.full((4, 4), np.nan)
    s[0, 1], s[2, 1], s[3, 1] = 0.9, 0.8, 0.7
    s[0, 2], s[0, 3] = 0.2, 0.1
    assert set(top_k_edges(s, 3, per_node=0)) == {(0, 1), (2, 1), (3, 1)}
    assert set(top_k_edges(s, 3, per_node=1)) == {(0, 1), (0, 2), (0, 3)}
    assert set(top_k_edges(s, 4, per_node=1)) == {(0, 1), (0, 2), (0, 3), (2, 1)}


def test_statistical_graph_uses_only_given_users():
    rows = [(u, [(0, 1), (1, 1)]) for u in range(6)] + [(u, [(0, 0), (1, 0)]) for u in range(6, 12)]
    seqs = _seqs(rows)
    adj, info = statistical_graph(seqs, users=range(12), n_nodes=3, n_edges=5, min_support=1)
    assert info["n_users"] == 12 and adj[0, 1] == 1
    adj_h, _ = statistical_graph(seqs, users=range(12), n_nodes=3, n_edges=5, heldout=[1], min_support=1)
    assert adj_h.nnz == 0
    _, info_sub = statistical_graph(seqs, users=range(6), n_nodes=3, n_edges=5, min_support=1)
    assert info_sub["n_users"] == 6


def test_random_and_dense_graph():
    g = random_graph(20, 50, seed=1)
    assert g.nnz == 50 and g.diagonal().sum() == 0
    assert (random_graph(20, 50, seed=1) != g).nnz == 0
    d = dense_graph(4)
    assert d.nnz == 12 and d.diagonal().sum() == 0


def test_heldout_only_from_nodes_with_interactions():
    nodes = pd.DataFrame({"node_id": range(10), "n_interactions": [0, 5, 5, 5, 5, 0, 5, 5, 5, 5]})
    h = choose_heldout_concepts(nodes, frac=0.25, seed=0)
    assert len(h) == 2 and not set(h) & {0, 5}


def test_compare_graphs():
    a = build_adjacency([(0, 1), (2, 1)], 3)
    ref = build_adjacency([(0, 1), (1, 2)], 3)
    c = compare_graphs(a, ref)
    assert c["shared_directed"] == 1 and c["reversed"] == 1
    assert c["jaccard_directed"] == pytest.approx(1 / 3)
    assert c["jaccard_undirected"] == pytest.approx(1.0)
