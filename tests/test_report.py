import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report"))

from dien_so_lieu import cite  # noqa: E402
from md_ast import ieee, parse  # noqa: E402
from tex import esc  # noqa: E402


def test_cite_numbers_by_first_appearance():
    bib = {"a": "A.", "b": "B.", "c": "C."}
    md = "x [@b] y [@a; @b] z\n\n<!-- TAI_LIEU -->"
    out = cite(md, bib)
    assert out.startswith("x [1] y [2], [1] z")
    assert "[1] B.\n\n[2] A." in out and "C." not in out


def test_ieee_blocks():
    md = ("# T\n\n## Tóm tắt\n\nNội dung.\n\n**Từ khóa—**a, b.\n\n## 1. Giới thiệu\n\n**Bảng I.** Chú thích.\n\n"
          "| x | y |\n|---|---|\n| 1 | 2 |\n\n## Tài liệu tham khảo\n\n[1] Tác giả, *Tạp chí*.\n\n[2] Khác.")
    blocks = ieee(parse(md))
    types = [b["type"] for b in blocks]
    assert types == ["h", "abstract", "keywords", "h", "table", "refs"]
    assert blocks[4]["caption"][0]["text"] == "Chú thích."
    assert len(blocks[5]["items"]) == 2 and blocks[5]["items"][0][0]["text"] == "Tác giả, "


def test_esc_math_symbols():
    assert esc("10⁻³ ≥ a_b") == "10$^{-3}$ $\\geq$ a\\_b"
