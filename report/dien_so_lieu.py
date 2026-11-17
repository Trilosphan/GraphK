"""Điền bảng và số liệu từ report/data/ket_qua.json vào các vị trí đánh dấu trong docs/BAO_CAO.md.

Vị trí đánh dấu (chú thích HTML, đứng riêng một dòng):
  <!-- BANG1 -->  kết quả chính        <!-- BANG2 -->  nghiên cứu cắt bỏ
  <!-- BANG3 -->  khái niệm giữ lại    <!-- BANG4 -->  AUC theo nhóm khái niệm
  <!-- DELONG --> kiểm định ghép cặp   <!-- CASE -->   tóm tắt case study
Trong câu: {{ten_so_lieu}}, ví dụ {{auc:GKT-Expert}}, {{auc_giu_lai:DKT}}, {{delta:GKT-Expert:GKT-Stat}}.

    python report/dien_so_lieu.py docs/BAO_CAO.md report/data/ket_qua.json build/BAO_CAO.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MAIN = ["GKT-Expert", "GKT-Stat", "GKT-Random", "DKT", "AKT"]


def num(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def pval(p: float) -> str:
    return "< 10⁻⁴" if p < 1e-4 else num(p, 4)


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def by(rows: list[dict], key: str = "mô hình") -> dict:
    return {r[key]: r for r in rows}


def bang1(k: dict) -> str:
    b = by(k["bang1"])
    best = max((b[m]["AUC kiểm tra"] for m in MAIN if m in b), default=None)
    rows = []
    for m in MAIN:
        if m not in b:
            continue
        r = b[m]
        auc = num(r["AUC kiểm tra"])
        rows.append([m, str(r["epoch"]), num(r["AUC kiểm định"]), f"**{auc}**" if r["AUC kiểm tra"] == best else auc,
                     num(r["ACC kiểm tra"]), num(r["RMSE kiểm tra"]), num(r["phút huấn luyện"], 0)])
    return table(["Mô hình", "Epoch", "AUC kiểm định", "AUC kiểm tra", "ACC", "RMSE", "Phút huấn luyện"], rows)


def bang2(k: dict) -> str:
    if not k["bang2"]:
        return "*Nghiên cứu cắt bỏ chưa có kết quả.*"
    return table(["Biến thể", "Epoch", "AUC kiểm định", "AUC kiểm tra", "AUC khái niệm giữ lại"],
                 [[r["mô hình"], str(r["epoch"]), num(r["AUC kiểm định"]), num(r["AUC kiểm tra"]), num(r["AUC khái niệm giữ lại"])]
                  for r in k["bang2"]])


def bang3(k: dict) -> str:
    b = by(k["bang3"])
    return table(["Mô hình", "AUC toàn tập", "AUC khái niệm còn lại", "AUC khái niệm giữ lại", "Chênh lệch (còn lại − giữ lại)"],
                 [[m, num(b[m]["AUC kiểm tra"]), num(b[m]["AUC khái niệm còn lại"]), num(b[m]["AUC khái niệm giữ lại"]),
                   num(b[m]["chênh lệch (còn lại − giữ lại)"])] for m in MAIN if m in b])


def bang4(k: dict) -> str:
    rows = k["bang4"]
    models = [m for m in MAIN if rows and m in rows[0]]
    counts = k.get("so_du_doan_theo_nhom", {})
    out = []
    for r in rows:
        label = r["nhóm"] + (f" ({counts[r['nhóm']]:,} dự đoán)".replace(",", ".") if r["nhóm"] in counts else "")
        out.append([label] + [num(r.get(m)) for m in models])
    return table(["Nhóm khái niệm"] + models, out)


def delong(k: dict) -> str:
    return table(["Tập", "Mô hình 1", "Mô hình 2", "AUC 1", "AUC 2", "Hiệu", "Sai số chuẩn", "p"],
                 [[r["tập"], r["mô hình 1"], r["mô hình 2"], num(r["auc_1"]), num(r["auc_2"]), num(r["delta"]),
                   num(r["se"], 5), pval(r["p_value"])] for r in k["kiem_dinh"]])


def case(k: dict) -> str:
    c = k["case_study"]
    names = dict(zip(c["focus"], c["focus_names"]))
    chain = "; ".join(f"`{names.get(a, a)}` → `{names.get(b, b)}`" for a, b in c["chain"])
    acc = ", ".join(f"{m}: {num(v, 2)}" for m, v in c["accuracy_next"].items())
    return (f"- Người học `{c['user_id']}`, đoạn chuỗi dài {len(c['q'])} bước (đoạn số {c['segment']} của tập kiểm tra).\n"
            f"- Cạnh tiên quyết chuyên gia xuất hiện trong đoạn: {chain}.\n"
            f"- Tỉ lệ dự đoán đúng bước kế tiếp (ngưỡng 0,5) trong đoạn: {acc}.")


def scalar(k: dict, expr: str) -> str:
    parts = expr.split(":")
    b1, b3 = by(k["bang1"]), by(k["bang3"])
    if parts[0] == "auc":
        return num(b1[parts[1]]["AUC kiểm tra"])
    if parts[0] == "auc_giu_lai":
        return num(b3[parts[1]]["AUC khái niệm giữ lại"])
    if parts[0] == "auc_con_lai":
        return num(b3[parts[1]]["AUC khái niệm còn lại"])
    if parts[0] == "delta":
        tap = parts[3] if len(parts) > 3 else "toàn bộ"
        r = next(r for r in k["kiem_dinh"] if r["mô hình 1"] == parts[1] and r["mô hình 2"] == parts[2] and r["tập"] == tap)
        return num(r["delta"])
    raise KeyError(expr)


def fill(md: str, k: dict) -> str:
    blocks = {"BANG1": bang1, "BANG2": bang2, "BANG3": bang3, "BANG4": bang4, "DELONG": delong, "CASE": case}
    for key, fn in blocks.items():
        md = md.replace(f"<!-- {key} -->", fn(k))
    return re.sub(r"\{\{([^}]+)\}\}", lambda m: scalar(k, m.group(1)), md)


if __name__ == "__main__":
    md = Path(sys.argv[1]).read_text(encoding="utf-8")
    k = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fill(md, k), encoding="utf-8")
    print("đã ghi", out)
