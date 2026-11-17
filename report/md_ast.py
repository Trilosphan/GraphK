"""Phân tích tập con Markdown dùng trong docs/BAO_CAO.md thành cây cấu trúc (AST) dạng JSON.

Khối hỗ trợ: tiêu đề (#..####), đoạn văn, danh sách gạch đầu dòng/đánh số (lồng tối đa 3 mức),
bảng (|...|), trích dẫn (> ...), hình (![chú thích](đường dẫn)), đường kẻ (---), chú thích HTML (bỏ qua).
Định dạng trong dòng: **đậm**, *nghiêng*, `mã`, [chữ](url).

    python report/md_ast.py docs/BAO_CAO.md > build/bao_cao.ast.json
"""

from __future__ import annotations

import json
import re
import sys

INLINE = re.compile(r"(\*\*(?P<b>.+?)\*\*|`(?P<c>[^`]+)`|\*(?P<i>[^*\s][^*]*?)\*|\[(?P<lt>[^\]]+)\]\((?P<lu>[^)]+)\))")


def inline(text: str) -> list[dict]:
    """Chuỗi -> danh sách đoạn chữ {text, b?, i?, code?, link?}."""
    runs, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            runs.append({"text": text[pos:m.start()]})
        if m.group("b") is not None:
            for r in inline(m.group("b")):
                runs.append({**r, "b": True})
        elif m.group("c") is not None:
            runs.append({"text": m.group("c"), "code": True})
        elif m.group("i") is not None:
            for r in inline(m.group("i")):
                runs.append({**r, "i": True})
        else:
            runs.append({"text": m.group("lt"), "link": m.group("lu")})
        pos = m.end()
    if pos < len(text):
        runs.append({"text": text[pos:]})
    return [r for r in runs if r["text"]]


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip().replace("\\|", "|") for c in cells]


def parse(md: str) -> list[dict]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = md.split("\n")
    blocks, i, para = [], 0, []

    def flush():
        if para:
            blocks.append({"type": "p", "runs": inline(" ".join(s.strip() for s in para))})
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if m := re.match(r"^(#{1,4})\s+(.*)$", s):
            flush()
            blocks.append({"type": "h", "level": len(m.group(1)), "runs": inline(m.group(2))})
            i += 1; continue
        if s == "---":
            flush(); blocks.append({"type": "hr"}); i += 1; continue
        if m := re.match(r"^!\[(.*)\]\((.+)\)$", s):
            flush(); blocks.append({"type": "figure", "caption": inline(m.group(1)), "path": m.group(2)})
            i += 1; continue
        if s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip()); i += 1
            blocks.append({"type": "quote", "lines": [inline(q) for q in quote if q]})
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            header = _split_row(rows[0])
            body = [_split_row(r) for r in rows[2:]] if len(rows) > 1 and re.match(r"^\|[\s:\-|]+\|$", rows[1].strip()) else [_split_row(r) for r in rows[1:]]
            blocks.append({"type": "table", "header": [inline(c) for c in header],
                           "rows": [[inline(c) for c in r] for r in body]})
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            flush()
            items = []
            while i < len(lines) and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip() and items)):
                ln = lines[i]
                if m := re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln):
                    items.append({"level": len(m.group(1).expandtabs(2)) // 2, "ordered": m.group(2)[0].isdigit(),
                                  "runs": inline(m.group(3))})
                else:  # dòng tiếp nối của mục trước
                    items[-1]["runs"] += [{"text": " "}] + inline(ln.strip())
                i += 1
            blocks.append({"type": "list", "items": items})
            continue
        para.append(s)
        i += 1
    flush()
    return blocks


if __name__ == "__main__":
    text = open(sys.argv[1], encoding="utf-8").read()
    json.dump(parse(text), sys.stdout, ensure_ascii=False, indent=1)
