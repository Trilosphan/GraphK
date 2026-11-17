import numpy as np
import pandas as pd

from graphk.preprocess import (
    build_adjacency,
    build_node_index,
    build_prerequisite_edges,
    build_sequences,
    graph_stats,
    is_dag,
    load_exercise_table,
    load_problem_log,
    parse_prerequisites,
    sort_and_filter,
    split_users,
)


def _table():
    # a -> b -> c, a -> c; d cô lập; e tham chiếu tiên quyết không tồn tại; f tự tham chiếu
    return pd.DataFrame({
        "name": ["a", "b", "c", "d", "e", "f"],
        "prerequisites": [np.nan, "a", "a, b, b", "", "ghost", "f"],
    })


def test_load_exercise_table_dedup(tmp_path):
    path = tmp_path / "ex.csv"
    pd.DataFrame({
        "name": ["a", "b", "a"],
        "prerequisites": ["", "a", "b"],
        "creation_date": ["2014-06-20 UTC", "2014-01-01 UTC", "2014-06-09 UTC"],
    }).to_csv(path, index=False)
    table = load_exercise_table(path)
    assert table["name"].tolist() == ["a", "b"]
    assert table.loc[table.name == "a", "creation_date"].item() == "2014-06-20 UTC"


def test_parse_prerequisites():
    assert parse_prerequisites(np.nan) == []
    assert parse_prerequisites("") == []
    assert parse_prerequisites("x, y ,z") == ["x", "y", "z"]


def test_edges_direction_and_report():
    table = _table()
    idx = build_node_index(table["name"])
    rep = build_prerequisite_edges(table, idx)
    assert set(rep.edges) == {(idx["a"], idx["b"]), (idx["a"], idx["c"]), (idx["b"], idx["c"])}
    assert rep.missing_refs == [("ghost", "e")]
    assert rep.self_loops == 1
    assert rep.duplicates == 1


def test_adjacency_and_stats():
    table = _table()
    idx = build_node_index(table["name"])
    adj = build_adjacency(build_prerequisite_edges(table, idx).edges, len(idx))
    assert adj[idx["a"], idx["b"]] == 1 and adj[idx["b"], idx["a"]] == 0
    s = graph_stats(adj)
    assert s["n_nodes"] == 6 and s["n_edges"] == 3
    assert s["n_isolated"] == 3  # d, e, f
    assert s["n_roots"] == 1 and s["n_leaves"] == 1
    assert s["n_weak_components"] == 4
    assert s["is_dag"]


def test_cycle_detected():
    adj = build_adjacency([(0, 1), (1, 2), (2, 0), (2, 3)], 4)
    assert not is_dag(adj)
    assert graph_stats(adj)["cycles"] == [[0, 1, 2]]


def test_load_sort_filter_sequences(tmp_path):
    log = pd.DataFrame({
        "user_id": [2, 1, 1, 1, 2, 3, 1, 2],
        "exercise": ["a", "b", "a", "zzz", "c", "a", "c", "b"],
        "problem_type": 0,
        "time_done": [30, 20, 10, 15, 10, 5, 20, 20],
        "correct": ["true", "false", "true", "true", "false", "true", "TRUE", "true"],
    })
    path = tmp_path / "log.csv"
    log.to_csv(path, index=False)
    idx = {"a": 0, "b": 1, "c": 2}
    df, info = load_problem_log(path, idx, chunksize=3)
    assert info["n_rows_raw"] == 8 and info["n_rows_unknown_exercise"] == 1
    df = sort_and_filter(df, min_interactions=3)
    assert set(df["user_id"]) == {1, 2}  # người học 3 chỉ có 1 tương tác
    # người học 1: t=10 (a), t=20 (b, rồi c theo thứ tự trong tệp)
    assert df.loc[df.user_id == 1, "node_id"].tolist() == [0, 1, 2]
    assert df.loc[df.user_id == 1, "correct"].tolist() == [1, 0, 1]

    seq = build_sequences(df, max_len=2, min_len=2)
    # mỗi người học có 3 tương tác -> một đoạn dài 2, đoạn dư dài 1 bị bỏ
    assert seq["offsets"].tolist() == [0, 2, 4]
    assert seq["seq_user"].tolist() == [1, 2]


def test_split_users_disjoint():
    s = split_users(np.arange(100), n_folds=5, test_frac=0.2, seed=0)
    assert (s["split"] == "test").sum() == 20
    assert s["user_id"].is_unique
    assert sorted(s.loc[s.split != "test", "split"].value_counts().tolist()) == [16] * 5
