"""Sinh báo cáo LaTeX (XeLaTeX, tiếng Việt) từ AST của docs/BAO_CAO.md.

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


def esc(s: str) -> str:
    return "".join(SPECIAL.get(ch, ch) for ch in s)


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
UNNUMBERED = ("Tóm tắt", "Phụ lục", "Tài liệu tham khảo", "Kế hoạch hình")


def heading(level: int, runs: list[dict]) -> str:
    text = plain(runs)
    numbered = bool(NUM.match(text)) and not text.startswith(UNNUMBERED)
    text = NUM.sub("", text)
    cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "paragraph")
    title = esc(text)
    if numbered:
        return f"\\{cmd}{{{title}}}"
    return f"\\{cmd}*{{{title}}}\n\\addcontentsline{{toc}}{{{cmd}}}{{{title}}}"


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


def table_tex(b: dict) -> str:
    n = len(b["header"])
    # cột đầu co theo nội dung, các cột còn lại chia đều bề rộng
    spec = "l" + "X" * (n - 1) if n > 2 else "lX"
    head = " & ".join(r"\textbf{" + runs_tex(c) + "}" for c in b["header"])
    rows = [" & ".join(runs_tex(c) for c in (r + [[]] * (n - len(r)))[:n]) + r" \\" for r in b["rows"]]
    return ("\\begin{center}\\small\n\\begin{tabularx}{\\linewidth}{" + spec.replace("X", ">{\\raggedright\\arraybackslash}X") + "}\n"
            "\\toprule\n" + head + r" \\" + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{center}")


def figure_tex(b: dict, fig_dir: str) -> str:
    cap = re.sub(r"^Hình\s+\d+\.\s*", "", plain(b["caption"]))
    name = Path(b["path"]).name
    return ("\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{fig_dir}/{name}}}\n"
            f"\\caption{{{esc(cap)}}}\n\\end{{figure}}")


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{fontspec}
\setmainfont{DejaVuSerif}[Path=fonts/,Extension=.ttf,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]
\setsansfont{DejaVuSans}[Path=fonts/,Extension=.ttf,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique]
\setmonofont{DejaVuSansMono}[Path=fonts/,Extension=.ttf,Scale=0.88,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique]
\usepackage{graphicx,booktabs,tabularx,array,xcolor,enumitem,caption,float}
\usepackage[hidelinks]{hyperref}
\setlist{nosep,leftmargin=1.4em}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}
\renewcommand{\contentsname}{Mục lục}
\captionsetup{font=small,labelsep=period}
\linespread{1.1}
"""


def build(ast: list[dict], info: dict, fig_dir: str = "figures") -> str:
    title = next((plain(b["runs"]) for b in ast if b["type"] == "h" and b["level"] == 1), "Báo cáo")
    body = []
    for b in ast:
        t = b["type"]
        if t == "h" and b["level"] == 1:
            continue
        if t == "h":
            body.append(heading(b["level"], b["runs"]))
        elif t == "p":
            body.append(runs_tex(b["runs"]))
        elif t == "list":
            body.append(list_tex(b["items"]))
        elif t == "table":
            body.append(table_tex(b))
        elif t == "figure":
            body.append(figure_tex(b, fig_dir))
        elif t == "quote":
            body.append("\\begin{quote}\\small\\itshape\n" + "\\\\\n".join(runs_tex(l) for l in b["lines"]) + "\n\\end{quote}")
        elif t == "hr":
            body.append("\\medskip")
    title_page = (
        "\\begin{titlepage}\n\\centering\n"
        f"{{\\large {esc(info.get('truong', ''))}}}\\\\[0.3em]\n{{\\large {esc(info.get('khoa', ''))}}}\\\\[3cm]\n"
        f"{{\\large {esc(info.get('mon_hoc', ''))}}}\\\\[1em]\n"
        f"{{\\LARGE\\bfseries {esc(title)}}}\\\\[3cm]\n"
        f"\\begin{{tabular}}{{ll}}\nGiảng viên: & {esc(info.get('giang_vien', ''))}\\\\\n"
        f"Thực hiện: & {esc(info.get('tac_gia', ''))}\\\\\nLớp: & {esc(info.get('lop', ''))}\\\\\n\\end{{tabular}}\n\\vfill\n"
        f"{esc(info.get('dia_diem_ngay', ''))}\n\\end{{titlepage}}\n\\tableofcontents\n\\newpage\n")
    return PREAMBLE + "\\begin{document}\n" + title_page + "\n\n".join(body) + "\n\\end{document}\n"


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
    out.write_text(build(ast, info), encoding="utf-8")
    copy_assets(out.parent, Path(sys.argv[4]) if len(sys.argv) > 4 else Path("report/figures"))
    print("đã ghi", out)
