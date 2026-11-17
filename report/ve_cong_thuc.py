"""Dựng các công thức trình chiếu bằng LaTeX (mathtext của matplotlib, phông Computer Modern) thành ảnh PNG nền trong suốt.

    python report/ve_cong_thuc.py report/figures/cong_thuc

Mỗi công thức một tệp <khóa>.png (300 dpi), kèm kich_thuoc.json ghi bề rộng/cao (inch) ở cỡ chữ CO_CHU,
để build_pptx.js đặt ảnh đúng cỡ chữ mong muốn. Trong công thức chỉ dùng ký hiệu toán (không có chữ tiếng Việt).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
CO_CHU = 20  # điểm
MAU = "#1F2A44"  # dk1 của chủ đề slide

CONG_THUC = {
    "kt": r"P\,(a_{t+1}=1 \mid q_{t+1},\, q_{1:t},\, a_{1:t})",
    "av": r"\mathrm{AV}(A \rightarrow B) = P(B \mid A) - P(B)",
    "gkt_x": r"x_t = E_{\mathrm{resp}}\,[\,2i + a_t\,]",
    "gkt_self": r"m_i = f_{\mathrm{self}}\,([\,h_i \,;\, x_t\,])",
    "gkt_nb": r"m_j = w^{\mathrm{fwd}}_{ij}\, f_{\mathrm{fwd}}(s_j) + w^{\mathrm{bwd}}_{ij}\, f_{\mathrm{bwd}}(s_j)",
    "gkt_s": r"s_j = [\,h_i \,;\, x_t \,;\, h_j \,;\, e_j\,]",
    "gkt_ea": r"\tilde{m}_j = m_j - \sigma(W_e m_j) \odot m_j + \tanh(W_a m_j)",
    "gkt_gru": r"h_j \leftarrow \mathrm{GRU}(\tilde{m}_j,\, h_j)",
    "gkt_pred": r"P\,(a_{t+1}=1) = \sigma(W_o\, h_{q_{t+1}} + b)",
    "dkt_h": r"h_t = \mathrm{LSTM}(x_t,\, h_{t-1})",
    "dkt_pred": r"P\,(a_{t+1}=1) = \sigma(W_{q_{t+1}}\, h_t + b)",
}


def ve(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for key, tex in CONG_THUC.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=CO_CHU, color=MAU)
        path = out / f"{key}.png"
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        import struct
        w, h = struct.unpack(">II", path.read_bytes()[16:24])
        sizes[key] = {"w": round(w / 300, 4), "h": round(h / 300, 4)}
    (out / "kich_thuoc.json").write_text(json.dumps({"co_chu": CO_CHU, "cong_thuc": sizes}, indent=1), encoding="utf-8")
    return sizes


if __name__ == "__main__":
    print(ve(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report/figures/cong_thuc")))
