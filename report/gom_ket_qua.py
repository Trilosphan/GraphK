"""Gom số liệu cho báo cáo và slide về một tệp report/data/ket_qua.json, chép các hình vào report/figures/.

    python report/gom_ket_qua.py <thư mục analysis> <thư mục processed/junyi> report

Nguồn: analysis/tom_tat.json, analysis/case_study.json, analysis/*.png (notebook 05);
processed/junyi/stats.json, processed/junyi/graphs/graph_summary.csv (notebook 02, 03).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def main(analysis: Path, proc: Path, report: Path) -> None:
    tom_tat = json.loads((analysis / "tom_tat.json").read_text(encoding="utf-8"))
    stats = json.loads((proc / "stats.json").read_text(encoding="utf-8"))
    graphs = pd.read_csv(proc / "graphs" / "graph_summary.csv")
    g = graphs[graphs["fold"].isin(["fold0", "-"])]
    out = {
        "du_lieu": {
            "n_nguoi_hoc": stats["n_users"], "n_tuong_tac": stats["n_interactions"],
            "ti_le_dung": stats["correct_rate"], "n_doan": stats["n_sequences"],
            "do_dai_tb": stats["mean_sequence_length"], "n_nut": stats["graph_expert"]["n_nodes"],
            "n_nut_khong_tuong_tac": stats["n_nodes_without_interactions"],
            "do_thi_chuyen_gia": {k: v for k, v in stats["graph_expert"].items() if k != "cycles"},
        },
        "do_thi": g.round(4).to_dict(orient="records"),
        **{k: tom_tat[k] for k in ("bang1", "bang2", "bang3", "bang4", "kiem_dinh", "so_du_doan_theo_nhom")},
        "case_study": json.loads((analysis / "case_study.json").read_text(encoding="utf-8")),
    }
    (report / "data").mkdir(parents=True, exist_ok=True)
    (report / "data" / "ket_qua.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (report / "figures").mkdir(parents=True, exist_ok=True)
    for f in analysis.glob("hinh*.png"):
        shutil.copy(f, report / "figures" / f.name)
    print("đã ghi", report / "data" / "ket_qua.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
