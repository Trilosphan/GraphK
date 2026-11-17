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


def notebook(cells, gpu: bool = False):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    if gpu:
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"]["gpuType"] = "T4"
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

NB03 = notebook([
    md("""\
# 03 — Xây dựng các đồ thị đối chứng và tập khái niệm giữ lại

**Đầu vào:** kết quả của notebook 02 (`GraphK/processed/junyi/`).
**Đầu ra** (`GraphK/processed/junyi/graphs/`):

| Tệp | Nội dung |
|---|---|
| `heldout_concepts.csv` | tập khái niệm giữ lại (RQ3): không có cạnh thống kê nào chạm vào |
| `adj_stat_fold{k}.npz` | đồ thị thống kê của phần k, chỉ tính trên người học thuộc tập huấn luyện của phần k |
| `adj_stat_all.npz` | đồ thị thống kê tính trên mọi người học ngoài tập kiểm tra (dùng khi đánh giá cuối) |
| `adj_random_fold{k}.npz` | đồ thị ngẫu nhiên, cùng số cạnh với đồ thị chuyên gia (đối chứng âm) |
| `graph_summary.csv` | đặc trưng của từng đồ thị và mức trùng cạnh với đồ thị chuyên gia |
| `config.json` | cấu hình đã dùng |

Đồ thị đầy đủ (GKT-Dense) không cần lưu: hàm `graphk.graphs.dense_graph` sinh trực tiếp khi huấn luyện.

**Định nghĩa đồ thị thống kê (GKT-Stat):**
- Người học `u` thành thạo khái niệm `c` khi tỉ lệ trả lời đúng của `u` trên `c` không dưới ngưỡng `MASTERY_THRESHOLD`.
- Điểm có hướng của cặp A → B là giá trị gia tăng (added value), chỉ tính trên người học đã làm cả A và B:
  `AV(A → B) = P(thành thạo B | thành thạo A) − P(thành thạo B)`.
- Chỉ giữ cặp có ít nhất `MIN_SUPPORT` người học thành thạo A và đã làm B.
- Chọn đúng số cạnh của đồ thị chuyên gia (979): trước hết mỗi khái niệm nhận cạnh vào có AV lớn nhất, phần còn lại lấy theo AV toàn cục.

**Chống rò rỉ dữ liệu:** đồ thị của phần k không dùng người học của tập kiểm định k hay tập kiểm tra; không cạnh thống kê nào chạm vào khái niệm giữ lại.

Không cần GPU."""),
    code(SETUP),
    md("## 0. Cấu hình"),
    code("""\
from pathlib import Path

PROC = Path(DRIVE) / 'processed' / 'junyi'
OUT = PROC / 'graphs'

HELDOUT_FRAC = 0.15        # tỉ lệ khái niệm giữ lại (trong số khái niệm có tương tác)
MASTERY_THRESHOLD = 0.5    # ngưỡng tỉ lệ đúng để coi là thành thạo
MIN_SUPPORT = 50           # số người học tối thiểu thành thạo A và đã làm B
PER_NODE = 1               # số cạnh vào tối thiểu ưu tiên cho mỗi khái niệm
N_FOLDS = 5
SEED = 42"""),
    md("## 1. Đọc dữ liệu và chọn tập khái niệm giữ lại"),
    code("""\
import numpy as np
import pandas as pd
from graphk.graphs import load_processed, choose_heldout_concepts, edge_set

d = load_processed(PROC)
nodes, adj_expert = d['nodes'], d['adj_expert']
N, E = len(nodes), adj_expert.nnz
heldout = choose_heldout_concepts(nodes, frac=HELDOUT_FRAC, seed=SEED)
held = set(heldout.tolist())
touching = sum(1 for a, b in edge_set(adj_expert) if a in held or b in held)
print(f'Số nút: {N} | số cạnh chuyên gia: {E}')
print(f'Khái niệm giữ lại: {len(heldout)} | cạnh chuyên gia chạm vào khái niệm giữ lại: {touching}')"""),
    md("## 2. Đồ thị thống kê theo từng phần"),
    code("""\
from graphk.graphs import statistical_graph, fold_users, compare_graphs
from graphk.preprocess import graph_stats

stat_graphs, rows = {}, []
for fold in [*range(N_FOLDS), None]:
    users = fold_users(d['splits'], fold)
    adj, info = statistical_graph(d['sequences'], users, N, E, heldout=heldout,
                                  threshold=MASTERY_THRESHOLD, min_support=MIN_SUPPORT, per_node=PER_NODE)
    name = 'all' if fold is None else f'fold{fold}'
    stat_graphs[name] = adj
    rows.append({'graph': 'stat', 'fold': name, **info, **graph_stats(adj), **compare_graphs(adj, adj_expert)})
    print(name, '| người học:', info['n_users'], '| AV từ', round(info['av_min'], 3), 'đến', round(info['av_max'], 3))"""),
    md("## 3. Đồ thị ngẫu nhiên và đồ thị chuyên gia (để so sánh)"),
    code("""\
from graphk.graphs import random_graph

random_graphs = {f'fold{k}': random_graph(N, E, seed=SEED + k) for k in range(N_FOLDS)}
for name, adj in random_graphs.items():
    rows.append({'graph': 'random', 'fold': name, 'n_edges': adj.nnz, **graph_stats(adj), **compare_graphs(adj, adj_expert)})
rows.append({'graph': 'expert', 'fold': '-', **graph_stats(adj_expert), **compare_graphs(adj_expert, adj_expert)})

cols = ['graph', 'fold', 'n_users', 'n_edges', 'n_isolated', 'n_weak_components', 'largest_component',
        'is_dag', 'jaccard_directed', 'jaccard_undirected', 'shared_directed', 'reversed']
summary = pd.DataFrame(rows)[cols]
summary"""),
    md("""\
## 4. Phân tích: đồ thị chuyên gia có khớp với tương quan thống kê không?

So sánh phân phối giá trị gia tăng (AV) của ba nhóm cặp khái niệm, tính trên tập huấn luyện của phần 0:
- cạnh chuyên gia theo đúng chiều (tiên quyết → phụ thuộc);
- cạnh chuyên gia theo chiều ngược;
- mọi cặp đủ hỗ trợ.

Nếu AV của cạnh chuyên gia không cao hơn mặt bằng chung, thì tri thức chuyên gia và tri thức thống kê là hai nguồn thông tin khác nhau. Đây là cơ sở để diễn giải RQ2."""),
    code("""\
from graphk.graphs import interactions_of_users, mastery_matrices, added_value_scores

u, q, c = interactions_of_users(d['sequences'], fold_users(d['splits'], 0))
att, mas = mastery_matrices(u, q, c, N, MASTERY_THRESHOLD)
av, _ = added_value_scores(att, mas, MIN_SUPPORT)
ex = list(edge_set(adj_expert))
groups = {
    'chuyên gia (xuôi)': np.array([av[a, b] for a, b in ex]),
    'chuyên gia (ngược)': np.array([av[b, a] for a, b in ex]),
    'mọi cặp đủ hỗ trợ': av[np.isfinite(av)],
}
av_table = pd.DataFrame({k: {'số cặp đủ hỗ trợ': int(np.isfinite(v).sum()),
                             'trung vị AV': float(np.nanmedian(v)),
                             'tứ phân vị 75%': float(np.nanpercentile(v, 75))} for k, v in groups.items()}).T
av_table"""),
    md("""\
**Kết quả kiểm tra trên máy (17/11/2026):**
- Chỉ 637/979 cạnh chuyên gia có đủ hỗ trợ để tính AV.
- Trung vị AV: cạnh chuyên gia theo chiều xuôi 0,020; theo chiều ngược 0,022; mọi cặp 0,013. Cạnh chuyên gia có tương quan thống kê cao hơn mặt bằng chung một chút, nhưng **dữ liệu hành vi không phân biệt được chiều tiên quyết**.
- Đồ thị thống kê trùng với đồ thị chuyên gia khoảng 1% cạnh (Jaccard có hướng ≈ 0,01). Đây là hai nguồn tri thức gần như tách rời, nên việc so sánh GKT-Expert với GKT-Stat (RQ2) có ý nghĩa.
- Đồ thị thống kê có khoảng 330 nút cô lập, trong đó 221 nút buộc phải cô lập: 108 khái niệm giữ lại và 113 khái niệm không có tương tác. Đồ thị chuyên gia chỉ có 59 nút cô lập. Khác biệt về độ phủ này cần được nêu khi diễn giải kết quả."""),
    md("## 5. Lưu kết quả lên Drive"),
    code("""\
import json
import scipy.sparse as sp

OUT.mkdir(parents=True, exist_ok=True)
nodes.loc[nodes['node_id'].isin(heldout), ['node_id', 'exercise_name', 'topic', 'area', 'n_interactions']] \\
    .to_csv(OUT / 'heldout_concepts.csv', index=False)
for name, adj in stat_graphs.items():
    sp.save_npz(OUT / f'adj_stat_{name}.npz', adj)
for name, adj in random_graphs.items():
    sp.save_npz(OUT / f'adj_random_{name}.npz', adj)
summary.to_csv(OUT / 'graph_summary.csv', index=False)
(OUT / 'config.json').write_text(json.dumps({
    'heldout_frac': HELDOUT_FRAC, 'mastery_threshold': MASTERY_THRESHOLD, 'min_support': MIN_SUPPORT,
    'per_node': PER_NODE, 'n_folds': N_FOLDS, 'seed': SEED, 'n_heldout': int(len(heldout)), 'n_edges': int(E),
}, indent=2))
!ls -la {OUT}"""),
])

NB04 = notebook([
    md("""\
# 04 — Huấn luyện GKT và các mô hình cơ sở

**Đầu vào:** kết quả của notebook 02 và 03.
**Đầu ra:** `GraphK/runs/<tên thực nghiệm>/` gồm `config.json`, `last.pt` (điểm kiểm tra mỗi epoch), `best.pt`, `results.json`, `test_predictions.npz`.

**Giao thức:** ở phần k, mô hình được huấn luyện trên các phần còn lại, dừng sớm theo AUC trên tập kiểm định (phần k), rồi đánh giá trên tập kiểm tra chung. Tập kiểm tra gồm 38.375 người học không tham gia huấn luyện hay xây dựng đồ thị.

**Các thực nghiệm:**

| Tên | Mô hình | Đồ thị |
|---|---|---|
| GKT-Expert | GKT lan truyền hai chiều | tiên quyết chuyên gia |
| GKT-Stat | GKT lan truyền hai chiều | thống kê (của phần k) |
| GKT-Random | GKT lan truyền hai chiều | ngẫu nhiên, cùng số cạnh |
| DKT | LSTM | không dùng đồ thị |
| AKT | Transformer (cài đặt của pyKT) | không dùng đồ thị |

**Nghiên cứu cắt bỏ** (`RUN_ABLATIONS = True`): GKT-Expert với nút gốc ảo cho nút cô lập, và GKT-Expert chỉ lan truyền chiều xuôi.

GKT-Dense không có trong danh sách mặc định: với 835 nút, mỗi bước phải truyền thông điệp tới 834 nút, vượt bộ nhớ GPU T4 ở cỡ lô thông thường.

**Chế độ chạy:**
- `QUICK = False` (mặc định): huấn luyện đầy đủ, tối đa `MAX_EPOCHS = 10` epoch, dừng sớm sau 3 epoch không cải thiện. Nếu phiên Colab bị ngắt, chạy lại notebook: mô hình học tiếp từ điểm kiểm tra gần nhất.
- `QUICK = True`: chạy thử trên phần 0, mỗi mô hình 300 lô huấn luyện và 100 lô đánh giá, để kiểm tra lỗi và **ước lượng thời gian** cho một epoch đầy đủ.
- `RUN_ABLATIONS = True` (mặc định): chạy thêm hai thí nghiệm cắt bỏ sau các thực nghiệm chính.

**Cần GPU:** Runtime → Change runtime type → T4 GPU."""),
    code(SETUP),
    md("## 0. Kiểm tra GPU"),
    code("""\
import torch
!nvidia-smi --query-gpu=name,memory.total --format=csv
assert torch.cuda.is_available(), 'Chưa bật GPU: Runtime → Change runtime type → T4 GPU'
print('torch', torch.__version__)"""),
    md("## 1. Cấu hình"),
    code("""\
from pathlib import Path

PROC = Path(DRIVE) / 'processed' / 'junyi'
QUICK = False                 # True: chạy thử ước lượng thời gian; False: huấn luyện đầy đủ
FOLDS = [0]                   # chạy đầy đủ: [0, 1, 2, 3, 4]
SEED = 42
BATCH_SIZE = 128
# Giới hạn 10 epoch cho mọi mô hình: GKT-Expert chỉ tăng ~0,0005 AUC kiểm định mỗi epoch từ epoch 4,
# điều kiện dừng sớm (3 epoch không cải thiện) không kích hoạt, 20 epoch làm phần 0 kéo dài ~24 giờ T4.
MAX_EPOCHS = 10

GKT_KW = {'hidden': 32, 'emb': 32, 'dropout': 0.1, 'bidirectional': True, 'norm': 'row', 'isolated': 'self'}
# (mô hình, đồ thị, tham số mô hình, cấu hình riêng của thực nghiệm)
EXPERIMENTS = [
    ('gkt', 'expert', GKT_KW, {}),
    ('gkt', 'stat', GKT_KW, {}),
    ('gkt', 'random', GKT_KW, {}),
    ('dkt', 'none', {'emb': 64, 'hidden': 64, 'dropout': 0.1}, {}),
    # AKT: cơ chế chú ý cần nhiều ma trận 200×200 mỗi head; cỡ lô 128 vượt 15 GB của T4
    ('akt', 'none', {'d_model': 256, 'n_blocks': 4, 'dropout': 0.2}, {'batch_size': 64}),
]

# Nghiên cứu cắt bỏ (Bảng 2 của báo cáo), chạy trên phần 0 sau các thực nghiệm chính
RUN_ABLATIONS = True
ABLATIONS = [
    ('gkt', 'expert', {**GKT_KW, 'isolated': 'root'}, {'tag': 'root'}),          # nút gốc ảo cho nút cô lập
    ('gkt', 'expert', {**GKT_KW, 'bidirectional': False}, {'tag': 'unidir'}),    # chỉ lan truyền chiều xuôi
]
if RUN_ABLATIONS:
    EXPERIMENTS = EXPERIMENTS + ABLATIONS

RUNS = Path(DRIVE) / ('runs_thu' if QUICK else 'runs')
QUICK_LIMITS = {'max_epochs': 1, 'max_train_batches': 300, 'max_eval_batches': 100}"""),
    md("## 2. Huấn luyện"),
    code("""\
from graphk.graphs import load_processed
from graphk.train import Config, run

import traceback

data = load_processed(PROC)
results, failed = [], []
for fold in FOLDS:
    for model, graph, kw, extra in EXPERIMENTS:
        opts = {'batch_size': BATCH_SIZE, **(QUICK_LIMITS if QUICK else {'max_epochs': MAX_EPOCHS}), **extra}
        cfg = Config(proc_dir=str(PROC), runs_dir=str(RUNS), model=model, graph=graph, fold=fold, seed=SEED,
                     model_kwargs=kw, **opts)
        try:
            results.append(run(cfg, data))
        except Exception:
            traceback.print_exc()
            failed.append(cfg.name)
        torch.cuda.empty_cache()
print('Thực nghiệm lỗi:', failed or 'không có')"""),
    md("## 3. Tổng hợp"),
    code("""\
import math
import pandas as pd
from graphk.train import segment_indices

n_train_segments = len(segment_indices(
    data['sequences'], data['splits'].loc[(data['splits'].split != 'test') & (data['splits'].split != '0'), 'user_id']))
rows = []
for res in results:
    h = res['history'][-1]
    n_train_batches = math.ceil(n_train_segments / res['config']['batch_size'])
    n_batches = res['config']['max_train_batches'] or n_train_batches
    rows.append({
        'thực nghiệm': res['name'],
        'epoch': res['epochs'],
        'AUC kiểm định': res['best_val_auc'],
        'AUC kiểm tra': res['test']['all']['auc'],
        'AUC khái niệm giữ lại': res['test']['heldout']['auc'],
        'ACC kiểm tra': res['test']['all']['acc'],
        'cỡ lô': res['config']['batch_size'],
        'giây/lô': h['train_seconds'] / n_batches,
        'phút/epoch đầy đủ (ước tính)': h['train_seconds'] / n_batches * n_train_batches / 60,
    })
summary = pd.DataFrame(rows)
summary.to_csv(RUNS / 'tong_hop.csv', index=False)
print('Số đoạn huấn luyện của một epoch đầy đủ:', n_train_segments)
summary"""),
], gpu=True)

NB05 = notebook([
    md("""\
# 05 — Phân tích kết quả

**Đầu vào:** `GraphK/runs/` (notebook 04), `GraphK/processed/junyi/` (notebook 02, 03).
**Đầu ra:** `GraphK/analysis/` gồm các bảng (`.csv`), hình (`.png`) và `tom_tat.json` (số liệu cho báo cáo).

Các phân tích:
1. **Bảng 1** — AUC, ACC, RMSE trên tập kiểm tra của mọi thực nghiệm (RQ1, RQ2).
2. **Bảng 2** — nghiên cứu cắt bỏ: xử lý nút cô lập, lan truyền một chiều/hai chiều.
3. **Bảng 3** — AUC trên tập khái niệm giữ lại và trên các khái niệm còn lại (RQ3).
4. **Bảng 4** — AUC theo nhóm khái niệm: cô lập, bậc thấp, bậc cao trong đồ thị chuyên gia; khái niệm cô lập trong đồ thị thống kê.
5. **Kiểm định DeLong ghép cặp** cho hiệu AUC giữa các mô hình, trên toàn tập kiểm tra và trên khái niệm giữ lại. Kiểm định giả định các dự đoán độc lập (bỏ qua tương quan trong cùng người học), và với hàng triệu dự đoán thì hiệu rất nhỏ cũng có ý nghĩa thống kê. Vì vậy cần đọc kèm độ lớn hiệu AUC.
6. **Hình** (theo kế hoạch hình của dàn ý): Hình 2 so sánh cấu trúc đồ thị, Hình 3 đường cong học, Hình 4 AUC trên khái niệm giữ lại, Hình 5 case study. Hình 1 (sơ đồ kiến trúc) được vẽ bằng `scripts/ve_so_do_kien_truc.py`.
7. **Case study** — một người học trong tập kiểm tra làm bài trên một chuỗi tiên quyết; mức thành thạo dự đoán theo từng bước của GKT-Expert, GKT-Stat và DKT.

Không cần GPU."""),
    code(SETUP),
    md("## 0. Cấu hình"),
    code("""\
import json
from pathlib import Path

PROC = Path(DRIVE) / 'processed' / 'junyi'
RUNS = Path(DRIVE) / 'runs'
OUT = Path(DRIVE) / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
FOLD, SEED = 0, 42

name = lambda model, graph, tag='': f'{model}-{graph}-fold{FOLD}-seed{SEED}' + (f'-{tag}' if tag else '')
MAIN = {'GKT-Expert': name('gkt', 'expert'), 'GKT-Stat': name('gkt', 'stat'), 'GKT-Random': name('gkt', 'random'),
        'DKT': name('dkt', 'none'), 'AKT': name('akt', 'none')}
ABL = {'GKT-Expert (hai chiều, tự cập nhật)': name('gkt', 'expert'),
       'GKT-Expert + nút gốc ảo': name('gkt', 'expert', 'root'),
       'GKT-Expert một chiều': name('gkt', 'expert', 'unidir')}"""),
    md("## 1. Bảng 1 — Kết quả trên tập kiểm tra"),
    code("""\
import pandas as pd
from graphk.analysis import load_results, results_table

results = load_results(RUNS)
table = results_table(results)
label = {v: k for k, v in {**ABL, **MAIN}.items()}
table.insert(0, 'mô hình', table['thực nghiệm'].map(label))
table = table.dropna(subset=['mô hình']).sort_values('AUC kiểm tra', ascending=False)
table.to_csv(OUT / 'bang1_ket_qua.csv', index=False)
print('Thực nghiệm chưa có kết quả:', [k for k, v in MAIN.items() if v not in results] or 'không có')
table.round(4)"""),
    md("## 2. Bảng 2 — Nghiên cứu cắt bỏ"),
    code("""\
abl = table[table['thực nghiệm'].isin(ABL.values())].copy()
abl['mô hình'] = abl['thực nghiệm'].map({v: k for k, v in ABL.items()})
abl.to_csv(OUT / 'bang2_cat_bo.csv', index=False)
abl[['mô hình', 'epoch', 'AUC kiểm định', 'AUC kiểm tra', 'AUC khái niệm giữ lại']].round(4)"""),
    md("## 3. Bảng 3 — Khái niệm giữ lại (RQ3)"),
    code("""\
main = table[table['thực nghiệm'].isin(MAIN.values())].copy()
main['mô hình'] = main['thực nghiệm'].map({v: k for k, v in MAIN.items()})
b3 = main[['mô hình', 'AUC kiểm tra', 'AUC khái niệm còn lại', 'AUC khái niệm giữ lại']].copy()
b3['chênh lệch (còn lại − giữ lại)'] = b3['AUC khái niệm còn lại'] - b3['AUC khái niệm giữ lại']
b3.to_csv(OUT / 'bang3_khai_niem_giu_lai.csv', index=False)
b3.round(4)"""),
    md("## 4. Bảng 4 — AUC theo nhóm khái niệm"),
    code("""\
import numpy as np
import scipy.sparse as sp
from graphk.analysis import load_predictions, node_groups, auc_by_group

preds = {k: load_predictions(RUNS, v) for k, v in MAIN.items() if v in results}
groups = node_groups(sp.load_npz(PROC / 'adj_expert.npz'), sp.load_npz(PROC / 'graphs' / f'adj_stat_fold{FOLD}.npz'))
b4a = auc_by_group(preds, groups['nhóm chuyên gia'].to_numpy())
stat_iso = np.where(groups['cô lập trong đồ thị thống kê'], 'cô lập trong đồ thị thống kê', 'có cạnh trong đồ thị thống kê')
b4b = auc_by_group(preds, stat_iso)
b4 = pd.concat([b4a, b4b])
counts = {g: int(np.isin(next(iter(preds.values()))['q'], groups.index[(groups['nhóm chuyên gia'] == g)]).sum())
          for g in b4a.index}
b4.to_csv(OUT / 'bang4_nhom_khai_niem.csv')
print('Số dự đoán theo nhóm chuyên gia:', counts)
b4.round(4)"""),
    md("## 5. Kiểm định DeLong ghép cặp"),
    code("""\
from graphk.analysis import pairwise_tests

heldout = pd.read_csv(PROC / 'graphs' / 'heldout_concepts.csv')['node_id'].to_numpy()
pairs = [('GKT-Expert', 'GKT-Stat'), ('GKT-Expert', 'GKT-Random'), ('GKT-Expert', 'DKT'),
         ('GKT-Expert', 'AKT'), ('GKT-Stat', 'GKT-Random'), ('DKT', 'AKT')]
t_all = pairwise_tests(preds, pairs).assign(tập='toàn bộ')
t_held = pairwise_tests(preds, pairs, mask_fn=lambda q: np.isin(q, heldout)).assign(tập='khái niệm giữ lại')
tests = pd.concat([t_all, t_held], ignore_index=True)
tests.to_csv(OUT / 'kiem_dinh_delong.csv', index=False)
tests[['tập', 'mô hình 1', 'mô hình 2', 'số dự đoán', 'auc_1', 'auc_2', 'delta', 'se', 'p_value']].round(5)"""),
    md("## 6. Hình"),
    code("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
for k, v in MAIN.items():
    if v in results:
        h = results[v]['history']
        ax.plot([e['epoch'] for e in h], [e['val']['all']['auc'] for e in h], marker='o', label=k)
ax.set_xlabel('epoch'); ax.set_ylabel('AUC kiểm định'); ax.set_title('Đường cong học (phần 0)')
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / 'hinh3_duong_cong_hoc.png', dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(b3))
ax.bar(x - 0.2, b3['AUC kiểm tra'], 0.4, label='toàn tập kiểm tra')
ax.bar(x + 0.2, b3['AUC khái niệm giữ lại'], 0.4, label='khái niệm giữ lại')
ax.set_xticks(x, b3['mô hình'], rotation=15)
lo = min(b3['AUC kiểm tra'].min(), b3['AUC khái niệm giữ lại'].min())
ax.set_ylim(lo - 0.02, None); ax.set_ylabel('AUC'); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title('AUC trên toàn tập và trên khái niệm giữ lại')
fig.tight_layout(); fig.savefig(OUT / 'hinh4_auc_giu_lai.png', dpi=150)
plt.show()"""),
    md("""\
## 7. Hình 2 — Đồ thị chuyên gia, thống kê và ngẫu nhiên"""),
    code("""\
from graphk.graphs import load_processed, fold_users, interactions_of_users, mastery_matrices, added_value_scores, edge_set
from graphk.analysis import plot_graph_comparison

data = load_processed(PROC)
adj_e = data['adj_expert']
adj_s = sp.load_npz(PROC / 'graphs' / f'adj_stat_fold{FOLD}.npz')
adj_r = sp.load_npz(PROC / 'graphs' / f'adj_random_fold{FOLD}.npz')
u, qq, cc = interactions_of_users(data['sequences'], fold_users(data['splits'], FOLD))
att, mas = mastery_matrices(u, qq, cc, len(data['nodes']), 0.5)
av, _ = added_value_scores(att, mas, 50)
ex = list(edge_set(adj_e))
av_groups = {'chuyên gia (xuôi)': np.array([av[a, b] for a, b in ex]),
             'chuyên gia (ngược)': np.array([av[b, a] for a, b in ex]),
             'cạnh thống kê': np.array([av[a, b] for a, b in edge_set(adj_s)]),
             'mọi cặp đủ hỗ trợ': av[np.isfinite(av)]}
plot_graph_comparison({'chuyên gia': adj_e, 'thống kê': adj_s, 'ngẫu nhiên': adj_r}, av_groups, OUT / 'hinh2_do_thi.png')
plt.show()"""),
    md("""\
## 8. Case study — Hình 5

Chọn tất định một đoạn chuỗi dài 25–60 bước trong tập kiểm tra có nhiều cạnh tiên quyết chuyên gia nhất (khái niệm tiên quyết xuất hiện trước khái niệm phụ thuộc, và kết quả trên khái niệm phụ thuộc có cả đúng lẫn sai). Với mỗi mô hình, `trace` cho xác suất trả lời đúng mọi khái niệm sau từng bước."""),
    code("""\
from graphk.analysis import select_case_study, load_trained, case_study_traces, plot_case_study
from graphk.train import segment_indices

test_users = data['splits'].loc[data['splits'].split == 'test', 'user_id']
seg, chain = select_case_study(data['sequences'], segment_indices(data['sequences'], test_users), adj_e)
q_seg = data['sequences']['questions'][data['sequences']['offsets'][seg]:data['sequences']['offsets'][seg + 1]].tolist()
in_chain = {c for e in chain for c in e}
focus = sorted(in_chain, key=q_seg.index)[:8]   # theo thứ tự xuất hiện đầu tiên trong đoạn
chain = [(int(a), int(b)) for a, b in chain if a in focus and b in focus]
names = data['nodes'].set_index('node_id').loc[focus, 'exercise_name'].tolist()
models = {k: load_trained(RUNS, MAIN[k], PROC, data) for k in ('GKT-Expert', 'GKT-Stat', 'DKT') if MAIN[k] in results}
case = case_study_traces(models, data['sequences'], seg, focus)
case.update({'segment': int(seg), 'user_id': int(data['sequences']['seq_user'][seg]), 'chain': chain,
             'focus_names': names,
             'q_names': data['nodes'].set_index('node_id').loc[case['q'], 'exercise_name'].tolist()})
y_next = np.array(case['r'][1:])
case['accuracy_next'] = {k: float(((np.array(v) >= 0.5) == y_next).mean()) for k, v in case['next_pred'].items()}
print('Đoạn', seg, '| người học', case['user_id'], '| độ dài', len(case['q']))
print('Cạnh tiên quyết trong đoạn:', [(names[focus.index(a)], names[focus.index(b)]) for a, b in chain])
print('Độ chính xác dự đoán bước kế tiếp:', case['accuracy_next'])
plot_case_study(case, names, OUT / 'hinh5_case_study.png')
(OUT / 'case_study.json').write_text(json.dumps(case, ensure_ascii=False, indent=1))
plt.show()"""),
    md("## 9. Tóm tắt cho báo cáo"),
    code("""\
import json

summary = {
    'bang1': table.round(5).to_dict(orient='records'),
    'bang2': abl.round(5).to_dict(orient='records'),
    'bang3': b3.round(5).to_dict(orient='records'),
    'bang4': b4.round(5).reset_index().to_dict(orient='records'),
    'so_du_doan_theo_nhom': counts,
    'kiem_dinh': tests.round(6).to_dict(orient='records'),
    'case_study': {k: case[k] for k in ('segment', 'user_id', 'chain', 'focus_names', 'accuracy_next')},
}
(OUT / 'tom_tat.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
!ls -la {OUT}"""),
])

NB06 = notebook([
    md("""\
# 06 — Xuất báo cáo LaTeX (PDF), Word (.docx) và slide seminar (.pptx)

**Đầu vào:** `GraphK/analysis/` (notebook 05), `GraphK/processed/junyi/`, nội dung `docs/BAO_CAO.md`, `report/noi_dung_slide.json`, `report/thong_tin.json` trong kho mã.
**Đầu ra** (`GraphK/bao_cao/`):

| Tệp | Nội dung |
|---|---|
| `BaoCao_GraphK.pdf`, `latex/` | báo cáo LaTeX (XeLaTeX, phông DejaVu) và mã nguồn |
| `BaoCao_GraphK.docx` | báo cáo Word |
| `Seminar_GraphK.pptx` | slide seminar có case study |
| `kiem_tra/` | ảnh từng trang/slide để rà soát hiển thị |

Không cần GPU. Lần đầu cài TeX Live và LibreOffice mất khoảng 5 phút."""),
    code(SETUP),
    md("## 1. Cài công cụ"),
    code("""\
!apt-get -qq update > /dev/null
!apt-get -qq install -y texlive-xetex texlive-latex-extra texlive-fonts-recommended libreoffice-writer libreoffice-impress poppler-utils > /dev/null
!cd {REPO}/report && npm install --silent pptxgenjs docx
!xelatex --version | head -1; node --version; soffice --version"""),
    md("## 2. Gom số liệu và vẽ hình"),
    code("""\
from pathlib import Path
ANALYSIS = Path(DRIVE) / 'analysis'
PROC = Path(DRIVE) / 'processed' / 'junyi'
OUT = Path(DRIVE) / 'bao_cao'
OUT.mkdir(parents=True, exist_ok=True)
%cd {REPO}
!python report/gom_ket_qua.py {ANALYSIS} {PROC} report
!python scripts/ve_so_do_kien_truc.py report/figures/hinh1_kien_truc.png
!ls report/figures"""),
    md("## 3. Báo cáo LaTeX → PDF"),
    code("""\
!python report/dien_so_lieu.py docs/BAO_CAO.md report/data/ket_qua.json build/BAO_CAO.md
!python report/md_ast.py build/BAO_CAO.md > build/bao_cao.ast.json
!python report/tex.py build/bao_cao.ast.json report/thong_tin.json build/latex/main.tex report/figures
!cd build/latex && xelatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null && xelatex -interaction=nonstopmode main.tex | tail -3
!cp build/latex/main.pdf {OUT}/BaoCao_GraphK.pdf && rm -rf {OUT}/latex && cp -r build/latex {OUT}/latex"""),
    md("## 4. Báo cáo Word và slide seminar"),
    code("""\
!NODE_PATH=report/node_modules node report/build_docx.js build/bao_cao.ast.json report/thong_tin.json report/figures {OUT}/BaoCao_GraphK.docx
!NODE_PATH=report/node_modules node report/build_pptx.js report {OUT}/Seminar_GraphK.pptx
# bỏ các mục thư mục trong tệp ZIP: Word/PowerPoint qua dịch vụ đám mây có thể từ chối mở tệp có các mục này
!zip -dq {OUT}/BaoCao_GraphK.docx '*/' ; zip -dq {OUT}/Seminar_GraphK.pptx '*/' ; echo xong"""),
    md("## 5. Kết xuất ảnh để rà soát hiển thị"),
    code("""\
CHECK = OUT / 'kiem_tra'
!rm -rf {CHECK} && mkdir -p {CHECK}
!cd {CHECK} && soffice --headless --convert-to pdf {OUT}/BaoCao_GraphK.docx > /dev/null && soffice --headless --convert-to pdf {OUT}/Seminar_GraphK.pptx > /dev/null
!cd {CHECK} && pdftoppm -jpeg -r 60 BaoCao_GraphK.pdf docx && pdftoppm -jpeg -r 80 Seminar_GraphK.pdf slide && pdftoppm -jpeg -r 60 {OUT}/BaoCao_GraphK.pdf tex
!ls {CHECK} | head -80"""),
])


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in (("01_tai_du_lieu.ipynb", NB01), ("02_tien_xu_ly.ipynb", NB02), ("03_do_thi.ipynb", NB03), ("04_huan_luyen.ipynb", NB04), ("05_phan_tich.ipynb", NB05), ("06_xuat_bao_cao.ipynb", NB06)):
        nbf.write(nb, out / name)
        print("đã ghi", out / name)
