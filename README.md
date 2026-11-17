# GraphK

Đồ án 8 môn Học máy: **truy vết tri thức dựa trên đồ thị (GKT) với đồ thị tiên quyết tường minh** trên dữ liệu Junyi Academy.

- Dàn ý đồ án: [`docs/DAN_Y.md`](docs/DAN_Y.md)
- Mã nguồn dùng chung: `src/graphk/`; kiểm thử: `tests/`

## Chạy trên Google Colab

Chạy các notebook theo thứ tự. Mỗi notebook tự gắn Google Drive, lấy mã mới nhất của kho này và lưu kết quả vào `MyDrive/GraphK/`.

| Notebook | Nội dung | Mở |
|---|---|---|
| 01 | Tải và giải nén dữ liệu Junyi (chạy một lần, cần khoảng 4 GB trên Drive) | [Mở trong Colab](https://colab.research.google.com/github/Trilosphan/GraphK/blob/main/notebooks/01_tai_du_lieu.ipynb) |
| 02 | Tiền xử lý: chỉ số nút, đồ thị tiên quyết, chuỗi tương tác, phân chia dữ liệu | [Mở trong Colab](https://colab.research.google.com/github/Trilosphan/GraphK/blob/main/notebooks/02_tien_xu_ly.ipynb) |
| 03 | Đồ thị thống kê theo từng phần, đồ thị ngẫu nhiên, tập khái niệm giữ lại | [Mở trong Colab](https://colab.research.google.com/github/Trilosphan/GraphK/blob/main/notebooks/03_do_thi.ipynb) |
| 04 | Huấn luyện GKT (Expert/Stat/Random), DKT, AKT; cần GPU T4 | [Mở trong Colab](https://colab.research.google.com/github/Trilosphan/GraphK/blob/main/notebooks/04_huan_luyen.ipynb) |

Các notebook được sinh từ `scripts/tao_notebook.py`. Khi cần sửa notebook, sửa script này rồi chạy lại `python scripts/tao_notebook.py`.

## Chạy kiểm thử trên máy

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" nbformat torch scikit-learn
.venv/bin/python -m pytest
```

Trên Mac Intel, PyTorch chỉ có bản 2.2 cho Python ≤ 3.12: tạo môi trường riêng bằng `uv venv .venv312 --python 3.12` rồi cài `torch==2.2.2 "numpy<2"`.

## Dữ liệu

Junyi Academy (bản 01/2015), nguồn gốc PSLC DataShop `datasetId=1198`. Dữ liệu không được dùng cho mục đích thương mại; khi sử dụng cần trích dẫn Chang, Hsu & Chen (2015), *Modeling Exercise Relationships in E-Learning: A Unified Approach*, EDM 2015.
