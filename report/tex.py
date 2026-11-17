"""Sinh bài báo LaTeX theo khuôn IEEE (lớp IEEEtran, hội nghị, hai cột; XeLaTeX, tiếng Việt) từ AST của docs/BAO_CAO.md.

Phông chữ: TeX Gyre Termes (tương đương Times, có dấu tiếng Việt). Các ký hiệu toán ngoài phông được viết bằng chế độ toán.

    python report/tex.py build/bao_cao.ast.json report/thong_tin.json build/latex/main.tex
    (cd build/latex && xelatex main.tex && xelatex main.tex)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


MATH = {"→": r"\rightarrow", "←": r"\leftarrow", "≈": r"\approx", "≥": r"\geq", "≤": r"\leq", "⊙": r"\odot",
        "∈": r"\in", "σ": r"\sigma", "⟨": r"\langle", "⟩": r"\rangle", "−": "-", "×": r"\times"}
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def esc(s: str) -> str:
    s = "".join(SPECIAL.get(ch, f"${MATH[ch]}$" if ch in MATH else ch) for ch in s)
    return re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "$^{" + m.group(0).translate(SUP) + "}$", s)


def runs_tex(runs: list[dict]) -> str:
    out = []
    for r in runs:
        t = esc(r["text"])
        if r.get("code"):
            t = r"\texttt{" + t + "}"
        if r.get("link"):
            t = r"\href{" + r["link"].replace("%", r"\%").replace("#", r"\#") + "}{" + t + "}"
        if r.get("i"):
            t = r"\emph{" + t + "}"
        if r.get("b"):
            t = r"\textbf{" + t + "}"
        out.append(t)
    return "".join(out)


def plain(runs: list[dict]) -> str:
    return "".join(r["text"] for r in runs)


NUM = re.compile(r"^(\d+(\.\d+)*\.?)\s+")


def heading(level: int, runs: list[dict]) -> str:
    """IEEEtran tự đánh số mục bằng số La Mã (I, II, …) và mục con bằng chữ (A, B, …)."""
    text = plain(runs)
    numbered = bool(NUM.match(text))
    title = esc(NUM.sub("", text))
    if not numbered and level == 2 and text.startswith("Phụ lục"):
        return "\\appendix\n\\section*{" + title + "}"
    cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "paragraph")
    return f"\\{cmd}{{{title}}}" if numbered else f"\\{cmd}*{{{title}}}"


def list_tex(items: list[dict]) -> str:
    out, stack = [], []
    for it in items:
        lvl = min(it["level"], 2)
        env = "enumerate" if it["ordered"] else "itemize"
        while len(stack) > lvl + 1:
            out.append(f"\\end{{{stack.pop()}}}")
        while len(stack) < lvl + 1:
            stack.append(env)
            out.append(f"\\begin{{{env}}}")
        out.append(r"\item " + runs_tex(it["runs"]))
    while stack:
        out.append(f"\\end{{{stack.pop()}}}")
    return "\n".join(out)


def wide_table(b: dict) -> bool:
    """Bảng nhiều cột hoặc ô dài trải hai cột (table*), như cách đặt bảng rộng trong bài IEEE."""
    longest = max((len(plain(c)) for r in [b["header"], *b["rows"]] for c in r), default=0)
    return len(b["header"]) >= 5 or longest > 45


def table_tex(b: dict) -> str:
    n = len(b["header"])
    wide = wide_table(b)
    env = "table*" if wide else "table"
    spec = "l" + ">{\\raggedright\\arraybackslash}X" * (n - 1)
    head = " & ".join(r"\textbf{" + runs_tex(c) + "}" for c in b["header"])
    rows = [" & ".join(runs_tex(c) for c in (r + [[]] * (n - len(r)))[:n]) + r" \\" for r in b["rows"]]
    cap = f"\\caption{{{runs_tex(b['caption'])}}}\n" if b.get("caption") else ""
    width = r"\textwidth" if wide else r"\columnwidth"
    # IEEE: chú thích bảng đặt trên bảng
    return (f"\\begin{{{env}}}[!t]\n\\centering\n{cap}\\footnotesize\n"
            f"\\begin{{tabularx}}{{{width}}}{{{spec}}}\n"
            "\\toprule\n" + head + r" \\" + "\n\\midrule\n" + "\n".join(rows) + f"\n\\bottomrule\n\\end{{tabularx}}\n\\end{{{env}}}")


def png_ratio(path: Path) -> float:
    import struct
    try:
        w, h = struct.unpack(">II", path.read_bytes()[16:24])
        return w / h
    except (OSError, struct.error):
        return 1.5


def figure_tex(b: dict, fig_dir: str, src_dir: Path) -> str:
    cap = re.sub(r"^Hình\s+\d+\.\s*", "", plain(b["caption"]))
    name = Path(b["path"]).name
    # hình rất ngang (tỉ lệ ≥ 2) trải hai cột
    env = "figure*" if png_ratio(src_dir / name) >= 2 else "figure"
    return (f"\\begin{{{env}}}[!t]\n\\centering\n"
            f"\\includegraphics[width=\\{'textwidth' if env == 'figure*' else 'columnwidth'}]{{{fig_dir}/{name}}}\n"
            f"\\caption{{{esc(cap)}}}\n\\end{{{env}}}")


PREAMBLE = r"""\documentclass[conference,a4paper]{IEEEtran}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{DejaVuSansMono}[Path=fonts/,Extension=.ttf,Scale=0.85,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique]
\usepackage{graphicx,booktabs,tabularx,array}
\usepackage[hidelinks]{hyperref}
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}
\renewcommand{\abstractname}{Tóm tắt}
\renewcommand{\IEEEkeywordsname}{Từ khóa}
\renewcommand{\refname}{Tài liệu tham khảo}
\renewcommand{\appendixname}{Phụ lục}
% tiếng Việt không có mẫu ngắt từ: cho phép giãn dòng để tránh tràn lề trong cột hẹp
\sloppy
\setlength{\emergencystretch}{3em}
"""


def build(ast: list[dict], info: dict, fig_dir: str = "figures", src_dir: Path = Path("report/figures")) -> str:
    title = next((plain(b["runs"]) for b in ast if b["type"] == "h" and b["level"] == 1), "Báo cáo")
    front, body = [], []
    for b in ast:
        t = b["type"]
        if t == "h" and b["level"] == 1:
            continue
        if t == "abstract":
            front.append("\\begin{abstract}\n" + runs_tex(b["runs"]) + "\n\\end{abstract}")
        elif t == "keywords":
            front.append("\\begin{IEEEkeywords}\n" + runs_tex(b["runs"]).strip() + "\n\\end{IEEEkeywords}")
        elif t == "h":
            body.append(heading(b["level"], b["runs"]))
        elif t == "p":
            body.append(runs_tex(b["runs"]))
        elif t == "list":
            body.append(list_tex(b["items"]))
        elif t == "table":
            body.append(table_tex(b))
        elif t == "figure":
            body.append(figure_tex(b, fig_dir, src_dir))
        elif t == "refs":
            body.append("\\begin{thebibliography}{" + "9" * len(str(len(b["items"]))) + "}\n"
                        + "\n".join(f"\\bibitem{{r{i + 1}}} " + runs_tex(it) for i, it in enumerate(b["items"]))
                        + "\n\\end{thebibliography}")
        elif t == "quote":
            body.append("\\begin{quote}\\small\\itshape\n" + "\\\\\n".join(runs_tex(l) for l in b["lines"]) + "\n\\end{quote}")
    author = (
        "\\author{\\IEEEauthorblockN{" + esc(info.get("tac_gia", "")) + "}\n"
        "\\IEEEauthorblockA{" + esc(info.get("khoa", "")) + ", " + esc(info.get("truong", "")) + "\\\\\n"
        + esc(info.get("mon_hoc", "")) + " -- Lớp " + esc(info.get("lop", "")) + "\\\\\n"
        "Giảng viên hướng dẫn: " + esc(info.get("giang_vien", "")) + "}}\n")
    return (PREAMBLE + "\\begin{document}\n\\title{" + esc(title) + "}\n" + author + "\\maketitle\n\n"
            + "\n\n".join(front) + "\n\n" + "\n\n".join(body) + "\n\\end{document}\n")


def copy_assets(out_dir: Path, figures: Path) -> None:
    """Chép phông DejaVu (đi kèm matplotlib) và các hình vào thư mục biên dịch."""
    import shutil
    import matplotlib
    src = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    (out_dir / "fonts").mkdir(parents=True, exist_ok=True)
    for f in src.glob("DejaVu*.ttf"):
        shutil.copy(f, out_dir / "fonts" / f.name)
    shutil.copy(src / "LICENSE_DEJAVU", out_dir / "fonts" / "LICENSE_DEJAVU")
    (out_dir / "figures").mkdir(exist_ok=True)
    for f in figures.glob("*.png"):
        shutil.copy(f, out_dir / "figures" / f.name)


if __name__ == "__main__":
    ast = json.load(open(sys.argv[1], encoding="utf-8"))
    info = json.load(open(sys.argv[2], encoding="utf-8"))
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    figures = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("report/figures")
    out.write_text(build(ast, info, src_dir=figures), encoding="utf-8")
    copy_assets(out.parent, figures)
    print("đã ghi", out)
