"""Sinh các notebook Colab trong notebooks/ từ mã nguồn này (chạy lại sau mỗi lần sửa).

    python scripts/tao_notebook.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """\
# Gắn Google Drive, lấy mã nguồn mới nhất của kho GraphK và cài gói graphk
from google.colab import drive
drive.mount('/content/drive')

import os, sys
DRIVE = '/content/drive/MyDrive/GraphK'
REPO = '/content/GraphK'
os.makedirs(DRIVE, exist_ok=True)
if not os.path.exists(REPO):
    !git clone -q https://github.com/Trilosphan/GraphK.git {REPO}
else:
    !git -C {REPO} pull -q
%pip install -q -e {REPO}
sys.path.insert(0, f'{REPO}/src')
!git -C {REPO} log --oneline -1"""


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    return nb


NB01 = notebook([
    md("""\
# 01 — Tải dữ liệu Junyi Academy

**Mục đích:** tải bộ dữ liệu Junyi Academy (bản 01/2015) về Google Drive, kiểm tra toàn vẹn và giải nén.

- **Nguồn gốc:** PSLC DataShop, `datasetId=1198`. Bản sao tải tự do trên máy chủ dữ liệu USTC (thư viện EduData sử dụng).
- **Điều kiện sử dụng:** không dùng cho mục đích thương mại; trích dẫn Chang, Hsu & Chen (2015), *Modeling Exercise Relationships in E-Learning: A Unified Approach*, EDM 2015.
- **Dung lượng:** tệp nén 1,3 GB, sau giải nén thêm 2,7 GB (chỉ lấy nhật ký gốc, bỏ qua tệp định dạng PSLC 9,7 GB). Cần khoảng **4 GB** trống trên Drive.
- **Thời gian:** chỉ cần chạy **một lần**. Nếu phiên Colab bị ngắt khi đang tải, chạy lại notebook, quá trình tải sẽ tiếp tục từ chỗ dừng.

Không cần GPU cho notebook này."""),
    code(SETUP),
    md("## 1. Công cụ giải nén"),
    code("""\
import shutil
if not shutil.which('unrar'):
    !apt-get -qq install -y unrar > /dev/null
print('unrar:', shutil.which('unrar'))"""),
    md("## 2. Tải tệp nén và kiểm tra toàn vẹn"),
    code("""\
from pathlib import Path
from graphk.data import download_junyi, sha256, JUNYI_SHA256

RAW = Path(DRIVE) / 'raw'
archive = download_junyi(RAW)
digest = sha256(archive)
(RAW / 'SHA256SUMS').write_text(f'{digest}  {archive.name}\\n')
print(archive, archive.stat().st_size, 'byte')
print('SHA256:', digest)
assert digest == JUNYI_SHA256, 'SHA256 khác giá trị đã ghi nhận — tệp tải về có thể hỏng hoặc máy chủ đã thay tệp'"""),
    md("## 3. Giải nén"),
    code("""\
from graphk.data import extract_junyi

paths = extract_junyi(archive, RAW / 'junyi')
for k, v in paths.items():
    print(f'{k:15s} {v}  ({v.stat().st_size / 1e9:.2f} GB)')"""),
    md("## 4. Xem nhanh bảng bài tập"),
    code("""\
import pandas as pd

table = pd.read_csv(paths['exercise_table'])
print('Số dòng (bài tập):', len(table))
print('Số bài còn hoạt động (live):', int(table['live'].astype(str).str.upper().eq('TRUE').sum()))
print('Số bài không có tiên quyết:', int(table['prerequisites'].isna().sum()))
table.head()"""),
])

NB02 = notebook([
    md("""\
# 02 — Tiền xử lý dữ liệu Junyi

**Đầu vào:** kết quả của notebook 01 trên Drive (`GraphK/raw/junyi/`).
**Đầu ra** (`GraphK/processed/junyi/`):

| Tệp | Nội dung |
|---|---|
| `nodes.csv` | ánh xạ `exercise_name → node_id`, kèm chủ đề, lĩnh vực, số tương tác |
| `edges_expert.csv`, `adj_expert.npz` | danh sách cạnh tiên quyết (nguồn → đích) và ma trận kề thưa |
| `sequences.npz` | chuỗi tương tác đã chia đoạn (≤ 200 bước) |
| `user_split.csv` | phân chia theo người học: tập kiểm tra 20%, phần còn lại chia 5 phần |
| `stats.json` | thống kê mô tả dữ liệu và đồ thị |

**Lựa chọn tiền xử lý** (đặt ở ô cấu hình bên dưới):
- tập nút của đồ thị;
- ngưỡng tối thiểu số tương tác của một người học;
- độ dài đoạn chuỗi tối đa;
- hạt giống ngẫu nhiên cho phân chia dữ liệu.

Không cần GPU. Nhật ký có khoảng 26 triệu dòng; nên chọn **High-RAM** nếu tài khoản có."""),
    code(SETUP),
    md("## 0. Cấu hình"),
    code("""\
from pathlib import Path

RAW = Path(DRIVE) / 'raw' / 'junyi'
OUT = Path(DRIVE) / 'processed' / 'junyi'

NODE_SET = 'table'      # 'table': mọi bài trong bảng bài tập | 'table_in_log': chỉ bài có tương tác
MIN_INTERACTIONS = 3    # bỏ người học có ít hơn ngưỡng này
MAX_LEN = 200           # độ dài đoạn chuỗi tối đa
SEED = 42"""),
    md("## 1. Bảng bài tập và chỉ số nút"),
    code("""\
from graphk.data import EXERCISE_TABLE, PROBLEM_LOG
from graphk.preprocess import load_exercise_table, build_node_index

table = load_exercise_table(next(RAW.rglob(EXERCISE_TABLE)))
print('Số bài trong bảng bài tập:', len(table))"""),
    md("""\
## 2. Đọc nhật ký tương tác

Đọc theo khối 2 triệu dòng, chỉ giữ 4 cột `user_id, exercise, time_done, correct`. Lần đọc đầu dùng chỉ số nút tạm gồm mọi bài trong bảng, để đếm số bài thực sự có tương tác."""),
    code("""\
from graphk.preprocess import load_problem_log

full_index = build_node_index(table['name'])
log, log_info = load_problem_log(next(RAW.rglob(PROBLEM_LOG)), full_index)
in_log = set(log['node_id'].unique())
print(f"Số dòng nhật ký: {log_info['n_rows_raw']:,}")
print(f"Số dòng thuộc bài không có trong bảng: {log_info['n_rows_unknown_exercise']:,}", log_info['unknown_exercises'][:10])
print('Số bài có tương tác:', len(in_log), '/', len(full_index))"""),
    md("""\
## 3. Chọn tập nút

Đề bài ghi **835 khái niệm**. Kiểm tra trên máy (17/11/2026) cho thấy con số này đến từ đâu:
- bảng bài tập có 837 dòng nhưng chỉ 835 tên khác nhau (`matrix_app_fruit_oil` và `matrix_mul_two` bị lặp);
- 776 bài còn hoạt động (live);
- 722 bài có tương tác trong nhật ký, nên 113 nút chỉ có trong đồ thị mà không có dữ liệu làm bài.

Mặc định `NODE_SET = 'table'` (835 nút) theo đề bài. Các nút không có tương tác vẫn tham gia lan truyền trên đồ thị. Ghi rõ lựa chọn này trong báo cáo."""),
    code("""\
names_in_log = {n for n, i in full_index.items() if i in in_log}
live = set(table.loc[table['live'].astype(str).str.upper().eq('TRUE'), 'name'])
print('Mọi bài trong bảng        :', len(full_index))
print('Bài còn hoạt động (live)  :', len(live))
print('Bài có tương tác          :', len(names_in_log))
print('Live và có tương tác      :', len(live & names_in_log))

node_names = list(full_index) if NODE_SET == 'table' else sorted(names_in_log)
node_index = build_node_index(node_names)
print('\\n=> NODE_SET =', NODE_SET, '| số nút =', len(node_index))"""),
    md("## 4. Đồ thị tiên quyết chuyên gia"),
    code("""\
from graphk.preprocess import build_prerequisite_edges, build_adjacency, graph_stats

rep = build_prerequisite_edges(table, node_index)
adj = build_adjacency(rep.edges, len(node_index))
g_stats = graph_stats(adj)
print('Tham chiếu tiên quyết không có trong tập nút:', len(rep.missing_refs), rep.missing_refs[:5])
print('Vòng tự thân bị bỏ:', rep.self_loops, '| cạnh trùng bị bỏ:', rep.duplicates)
for k, v in g_stats.items():
    if k != 'cycles':
        print(f'{k:20s} {v}')
names = list(node_index)
for cyc in g_stats['cycles']:
    print('Chu trình (thành phần liên thông mạnh):', [names[i] for i in cyc])"""),
    md("""\
**Kết quả kiểm tra trên máy (17/11/2026):**
- 835 nút, 979 cạnh.
- 59 nút cô lập, 69 thành phần liên thông yếu.
- Đồ thị **không phải DAG**: có một chu trình 3 bài về căn thức (`adding_and_subtracting_radicals` → `radical_multiplication_and_division` → `simplifying_radicals` → quay lại bài đầu).
- Hai bài tự ghi chính mình làm tiên quyết đã bị loại: `proportions_1`, `number_sense_length_l1`.

Lan truyền GNN không đòi hỏi đồ thị phải là DAG, nên giữ nguyên chu trình. Báo cáo cần nêu rõ đây là một điểm không nhất quán trong đồ thị chuyên gia."""),
    md("## 5. Chuỗi tương tác và phân chia dữ liệu"),
    code("""\
import numpy as np
from graphk.preprocess import sort_and_filter, build_sequences, split_users

# Ánh xạ lại node_id theo tập nút đã chọn
remap = np.full(len(full_index), -1, dtype=np.int32)
for name, i in full_index.items():
    if name in node_index:
        remap[i] = node_index[name]
log['node_id'] = remap[log['node_id'].to_numpy()]
log = log[log['node_id'] >= 0]

log = sort_and_filter(log, min_interactions=MIN_INTERACTIONS)
seqs = build_sequences(log, max_len=MAX_LEN, min_len=MIN_INTERACTIONS)
splits = split_users(seqs['seq_user'], n_folds=5, test_frac=0.2, seed=SEED)
lengths = np.diff(seqs['offsets'])
print(f"Người học: {log['user_id'].nunique():,} | tương tác: {len(log):,} | tỉ lệ đúng: {log['correct'].mean():.4f}")
print(f"Số đoạn: {len(lengths):,} | độ dài trung bình: {lengths.mean():.1f} | tương tác giữ lại: {lengths.sum():,}")
print(splits['split'].value_counts().sort_index())"""),
    md("## 6. Lưu kết quả lên Drive"),
    code("""\
import pandas as pd
from graphk.preprocess import save_processed

counts = np.bincount(log['node_id'], minlength=len(node_index))
nodes = pd.DataFrame({'node_id': range(len(node_index)), 'exercise_name': list(node_index)})
nodes = nodes.merge(table[['name', 'topic', 'area', 'live']], left_on='exercise_name', right_on='name', how='left').drop(columns='name')
nodes['n_interactions'] = counts

stats = {
    'config': {'node_set': NODE_SET, 'min_interactions': MIN_INTERACTIONS, 'max_len': MAX_LEN, 'seed': SEED},
    'log': {k: v for k, v in log_info.items() if k != 'unknown_exercises'},
    'n_users': int(log['user_id'].nunique()),
    'n_interactions': int(len(log)),
    'correct_rate': float(log['correct'].mean()),
    'n_sequences': int(len(lengths)),
    'mean_sequence_length': float(lengths.mean()),
    'n_nodes_without_interactions': int((counts == 0).sum()),
    'graph_expert': g_stats,
    'edges_missing_refs': len(rep.missing_refs),
}
save_processed(OUT, nodes=nodes, edges=rep.edges, adj=adj, sequences=seqs, splits=splits, stats=stats)
!ls -la {OUT}"""),
])


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in (("01_tai_du_lieu.ipynb", NB01), ("02_tien_xu_ly.ipynb", NB02)):
        nbf.write(nb, out / name)
        print("đã ghi", out / name)
