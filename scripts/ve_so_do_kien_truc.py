"""Vẽ Hình 1: sơ đồ kiến trúc GKT lan truyền hai chiều (một bước cập nhật).

    python scripts/ve_so_do_kien_truc.py report/figures/hinh1_kien_truc.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1F2A44"
PRE, CUR, DEP = "#4C9A8A", "#E07A3F", "#5B6FB5"   # tiên quyết, đang trả lời, phụ thuộc
OP = "#F1F3F8"


def box(ax, x, y, w, h, text, fc=OP, ec=INK, fs=10, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold" if bold else "normal")


def node(ax, x, y, label, color):
    ax.add_patch(plt.Circle((x, y), 0.32, fc=color, ec="white", lw=2, zorder=3))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=4)


def arrow(ax, p, q, color=INK, style="-|>", ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))


def main(out: str) -> None:
    fig, ax = plt.subplots(figsize=(12.9, 6.2))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # (a) Đồ thị tiên quyết quanh khái niệm đang trả lời
    ax.text(0.3, 5.85, "(a) Đồ thị tiên quyết chuyên gia", fontsize=11, weight="bold", color=INK)
    node(ax, 1.0, 4.6, "p₁", PRE)
    node(ax, 1.0, 3.0, "p₂", PRE)
    node(ax, 2.6, 3.8, "i", CUR)
    node(ax, 4.2, 4.6, "d₁", DEP)
    node(ax, 4.2, 3.0, "d₂", DEP)
    node(ax, 2.6, 2.0, "k", "#9AA3B5")
    for p in [(1.0, 4.6), (1.0, 3.0)]:
        arrow(ax, (p[0] + 0.3, p[1] - 0.12 if p[1] > 3.8 else p[1] + 0.12), (2.3, 3.8), color="#9AA3B5")
    for d in [(4.2, 4.6), (4.2, 3.0)]:
        arrow(ax, (2.9, 3.8), (d[0] - 0.3, d[1] - 0.12 if d[1] > 3.8 else d[1] + 0.12), color="#9AA3B5")
    ax.text(2.6, 1.55, "khái niệm không kề i:\nkhông cập nhật", ha="center", va="top", fontsize=8.5, color="#6B7385")
    ax.text(0.3, 0.45, "cạnh u → v: u là tiên quyết của v\n"
                       "p: tiên quyết của i  ·  d: phụ thuộc vào i", fontsize=8.5, color=INK)

    # (b) Một bước cập nhật
    ax.text(5.4, 5.85, "(b) Một bước cập nhật khi người học trả lời khái niệm i", fontsize=11, weight="bold", color=INK)
    box(ax, 6.4, 4.9, 1.9, 0.7, r"$(q_t = i,\ a_t)$", fc="#FBE7DA", ec=CUR, fs=11)
    box(ax, 6.4, 3.8, 1.9, 0.7, r"$x_t = E[2i + a_t]$", fs=11)
    box(ax, 6.4, 2.7, 1.9, 0.7, r"$s = [h_i\,;\,x_t]$", fs=11)
    arrow(ax, (6.4, 4.55), (6.4, 4.15))
    arrow(ax, (6.4, 3.45), (6.4, 3.05))

    box(ax, 8.9, 4.4, 2.3, 0.75, r"$f_{bwd}([s;h_p;e_p])$" + "\n→ tiên quyết p", fc="#DDEFEB", ec=PRE, fs=9.5)
    box(ax, 8.9, 2.7, 2.3, 0.75, r"$f_{self}(s)$" + "\n→ chính khái niệm i", fc="#FBE7DA", ec=CUR, fs=9.5)
    box(ax, 8.9, 1.0, 2.3, 0.75, r"$f_{fwd}([s;h_d;e_d])$" + "\n→ phụ thuộc d", fc="#E2E6F4", ec=DEP, fs=9.5)
    arrow(ax, (7.35, 2.85), (7.75, 4.25), rad=-0.15)
    arrow(ax, (7.35, 2.7), (7.75, 2.7))
    arrow(ax, (7.35, 2.55), (7.75, 1.15), rad=0.15)

    box(ax, 11.6, 2.7, 2.2, 0.75, "cổng xóa–thêm\n+ GRU", fs=9.5)
    for y in (4.4, 2.7, 1.0):
        arrow(ax, (10.05, y), (10.5, 2.7 + (y - 2.7) * 0.25), rad=0.0)
    box(ax, 11.6, 4.3, 2.2, 0.75, r"$h_j \leftarrow \mathrm{GRU}(m'_j, h_j)$" + "\nchỉ cập nhật i, p, d", fs=9)
    arrow(ax, (11.6, 3.08), (11.6, 3.92))
    box(ax, 11.6, 1.0, 2.2, 0.75, r"$P(a_{t+1}=1)$" + "\n" + r"$=\sigma(W h_{q_{t+1}} + b)$", fs=9.5)
    arrow(ax, (11.6, 2.32), (11.6, 1.38))
    ax.text(8.9, 0.25, "trọng số w: chia cho số lân cận cùng chiều (chuẩn hóa theo hàng)",
            ha="center", fontsize=8.5, color=INK)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("đã ghi", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/figures/hinh1_kien_truc.png")
