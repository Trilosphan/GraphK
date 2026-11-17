# Đồ án 8 — GKT trên đồ thị tiên quyết tường minh

> Môn Machine Learning · Mô hình trọng tâm: GKT (Graph-based Knowledge Tracing)
> Dữ liệu chính: Junyi Academy (835 khái niệm, đồ thị tiên quyết do chuyên gia xác nhận) · Dữ liệu bổ sung: MOOCCubeX
> Mức độ: Chuyên sâu · Môi trường thử nghiệm: Google Colab (GPU T4)

---

## 1. Giới thiệu

### 1.1. Trí tuệ nhân tạo trong giáo dục (AIED)
- AIED là gì: hệ thống dạy học thông minh (ITS), học tập thích ứng, mô hình người học.
- Vai trò của mô hình người học: ước lượng học sinh đã nắm khái niệm nào, từ đó gợi ý bài tiếp theo và lộ trình học.
- Bối cảnh nghiệp vụ: phòng đào tạo Junyi Academy đã có đồ thị tiên quyết do đội ngũ sư phạm xác nhận (ví dụ phải học Đạo hàm trước Tích phân). Các mô hình DKT/DKVMN ở đồ án 3–6 bỏ qua thông tin cấu trúc này.

### 1.2. Knowledge Tracing (KT)
- Định nghĩa: từ chuỗi tương tác `(q_1, a_1), …, (q_t, a_t)` của một học sinh, ước lượng trạng thái tri thức và dự đoán `P(a_{t+1} = 1 | q_{t+1}, lịch sử)`.
- Các họ mô hình chính: xác suất (BKT), học sâu theo chuỗi (DKT, DKVMN), chú ý (SAKT, AKT), đồ thị (GKT, GIKT, SKT).
- Hạn chế của mô hình chuỗi: mỗi khái niệm được xem độc lập hoặc phụ thuộc ẩn, không dùng quan hệ tiên quyết tường minh.

### 1.3. Định nghĩa bài toán
- **Đầu vào:**
  - Nhật ký làm bài Junyi: `user_id`, `exercise_name`, `correct`, `time_done`, ánh xạ `exercise_name → node_id ∈ [0, 834]`.
  - Đồ thị tiên quyết `G = (V, E)` với `|V| = 835`, cạnh có hướng (nguồn → đích), dựng thành ma trận kề thưa `A ∈ {0,1}^{835×835}`.
- **Đầu ra:** xác suất làm đúng ở bước kế tiếp cho từng khái niệm.
- **Câu hỏi nghiên cứu:**
  - **RQ1:** GKT dùng đồ thị chuyên gia có tốt hơn mô hình không dùng đồ thị (AKT, DKT) không?
  - **RQ2:** Tri thức chuyên gia và tri thức thống kê (đồ thị đồng xuất hiện) mỗi loại đóng góp bao nhiêu?
  - **RQ3:** Khi so sánh hai loại đồ thị có nguy cơ rò rỉ gián tiếp không, và cần thiết kế đánh giá thế nào cho công bằng?
  - **RQ4 (nếu đủ thời gian):** Mô hình có chuyển được sang MOOCCubeX không (zero-shot hoặc fine-tune nhẹ)?

### 1.4. Mục tiêu và đóng góp
- Dựng ma trận kề từ danh sách cạnh gốc và tùy biến công thức lan truyền GNN cho KT, có xử lý node cô lập.
- Đánh giá tách bạch ba biến thể: đồ thị chuyên gia, đồ thị thống kê, không đồ thị.
- Phân tích rò rỉ thông tin gián tiếp và đề xuất giao thức đánh giá trên nhóm khái niệm giữ lại.
- Pipeline tái lập được: một lệnh chạy từ dữ liệu thô đến bảng kết quả (rubric tăng trọng số phần kỹ thuật mã nguồn).

---

## 2. Các nghiên cứu liên quan

**Tiêu chí chọn bài:** công bố **từ 2020 trở lại đây**, trên tạp chí xếp hạng Q1–Q4 (Scimago/SJR) hoặc hội nghị hạng A/A* (CORE), có DOI hoặc trang kỷ yếu chính thức.
- Bài kinh điển trước 2020 (BKT 1994, DKT 2015, DKVMN 2017, GKT 2019) chỉ nhắc ở mục 1.2 và mục 3, **không** tính vào danh sách này.
- Hạng Q ghi trong bảng là hạng của tạp chí theo SJR. Trước khi nộp cần đối chiếu lại trên scimagojr.com theo đúng năm công bố.

Mỗi bài trình bày theo 5 ý: **phương pháp/giải thuật · dữ liệu · ưu · khuyết · đánh giá**. Mục 2.4 tổng hợp và nêu đề xuất.

### 2.1. Bài trên tạp chí (Q1)

**[J1] JKT — Song et al. (2021).** *JKT: A joint graph convolutional network based Deep Knowledge Tracing.* Information Sciences, 580, 510–523. DOI 10.1016/j.ins.2021.08.100.
- **Phương pháp:** dựng đồ thị bài tập–bài tập và khái niệm–khái niệm, kết hợp quan hệ bài tập–khái niệm, rồi đưa qua GCN kết hợp; phần chuỗi dùng DKT.
- **Dữ liệu:** 4 bộ dữ liệu thật (gồm các bộ ASSISTments).
- **Ưu:** nối được các bài tập thuộc những khái niệm khác nhau, diễn giải được qua nghiên cứu tình huống.
- **Khuyết:** đồ thị dựng từ thống kê và gán nhãn khái niệm, không dùng đồ thị tiên quyết chuyên gia; không phân biệt cạnh có hướng.

**[J2] Bi-CLKT — Song et al. (2022).** *Bi-CLKT: Bi-Graph Contrastive Learning based Knowledge Tracing.* Knowledge-Based Systems, 241, 108274. DOI 10.1016/j.knosys.2022.108274.
- **Phương pháp:** học tương phản hai tầng trên đồ thị con bài tập–bài tập: tầng node cho biểu diễn bài tập, tầng đồ thị cho biểu diễn khái niệm; lớp dự đoán là RNN hoặc mạng bộ nhớ.
- **Dữ liệu:** 4 bộ dữ liệu thật.
- **Ưu:** biểu diễn khái niệm tốt hơn khi dữ liệu thưa.
- **Khuyết:** đồ thị suy từ dữ liệu, dễ khớp với hành vi của tập đánh giá; khối lượng tính toán lớn.

**[J3] DGEKT — Cui et al. (2024).** *DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing.* ACM Transactions on Information Systems, 42(3), Article 78. DOI 10.1145/3638350.
- **Phương pháp:** hai đồ thị (siêu đồ thị bài tập–khái niệm và đồ thị có hướng chuyển tiếp giữa các tương tác), kết hợp bằng chưng cất tri thức trực tuyến.
- **Dữ liệu:** 3 bộ benchmark, so với 8 mô hình KT.
- **Ưu:** tận dụng cả quan hệ không thuần nhất lẫn quan hệ có hướng.
- **Khuyết:** đồ thị có hướng dựng từ chuyển tiếp thống kê, chưa từng đối chiếu với tri thức chuyên gia.

**[J4] HHSKT — Ni et al. (2023).** *HHSKT: A learner–question interactions based heterogeneous graph neural network model for knowledge tracing.* Expert Systems with Applications, 215, 119334. DOI 10.1016/j.eswa.2022.119334.
- **Phương pháp:** GNN không thuần nhất trên đồ thị học sinh–câu hỏi–khái niệm.
- **Ghi chú:** em chưa đọc được toàn văn; cần đọc bản gốc trước khi điền phần ưu–khuyết.

**[J5] Khảo sát — Abdelrahman, Wang & Nunes (2023).** *Knowledge Tracing: A Survey.* ACM Computing Surveys, 55(11), 1–37. DOI 10.1145/3569576.
- Dùng cho mục 1.2 (lịch sử, phân loại mô hình, các bộ dữ liệu benchmark) và để chỉ ra khoảng trống nghiên cứu.

**[J6] Khảo sát — Shen et al. (2024).** *A Survey of Knowledge Tracing: Models, Variants, and Applications.* IEEE Transactions on Learning Technologies, 17, 1898–1919. DOI 10.1109/TLT.2024.3383325.
- Phân loại 3 nhóm mô hình KT cơ bản và các biến thể; có thư viện mã nguồn mở đi kèm.

### 2.2. Bài trên hội nghị hạng A*/A

**[C1] AKT — Ghosh, Heffernan & Lan (2020).** *Context-Aware Attentive Knowledge Tracing.* KDD 2020. DOI 10.1145/3394486.3403282. — **mô hình đối chứng chính**
- **Phương pháp:** Transformer với chú ý suy giảm đơn điệu theo khoảng cách thời gian, embedding kiểu Rasch theo câu hỏi.
- **Ưu:** thuộc nhóm có AUC cao nhất trên nhiều benchmark.
- **Khuyết:** không dùng cấu trúc tiên quyết.

**[C2] SKT — Tong et al. (2020).** *Structure-based Knowledge Tracing: An Influence Propagation View.* IEEE ICDM 2020, 541–550.
- **Phương pháp:** lan truyền ảnh hưởng trên cấu trúc tri thức: quan hệ có hướng (tiên quyết) lan truyền từng phần, quan hệ vô hướng (tương tự) lan truyền đồng bộ.
- **Ưu:** gần nhất với đồ án vì dùng quan hệ tiên quyết có hướng; làm căn cứ cho cách tách hai chiều lan truyền ở mục 3.3 và cho `edge_type`.
- **Khuyết:** đồ thị tiên quyết chủ yếu suy từ dữ liệu; không tách riêng đóng góp của tri thức chuyên gia.

**[C3] GIKT — Yang et al. (2020).** *GIKT: A Graph-based Interaction Model for Knowledge Tracing.* ECML-PKDD 2020, LNCS 12457, 299–315. DOI 10.1007/978-3-030-67658-2_18.
- **Phương pháp:** GCN lan truyền embedding trên đồ thị câu hỏi–kỹ năng, mô hình hóa tương tác giữa trạng thái hiện tại, các bài đã làm có liên quan, câu hỏi đích và kỹ năng liên quan.
- **Đánh giá:** đạt SOTA trên 3 bộ dữ liệu, AUC tăng ít nhất khoảng 1%.

**[C4] simpleKT — Liu et al. (2023).** *simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing.* ICLR 2023.
- **Vai trò:** đối chứng mạnh bổ sung. Bài chỉ ra rằng nhiều mô hình phức tạp không hơn một baseline đơn giản khi đánh giá chuẩn hóa. Đây là lý do đồ án phải dùng giao thức chuẩn (pyKT).

**[C5] PSI-KT — Zhou et al. (2024).** *Predictive, scalable and interpretable knowledge tracing on structured domains.* ICLR 2024.
- **Phương pháp:** mô hình sinh phân cấp, suy ra đồng thời đặc điểm nhận thức của từng học sinh và đồ thị tiên quyết dùng chung, ước lượng độ mạnh tiên quyết từ dữ liệu làm bài.
- **Ưu:** diễn giải được, áp dụng được cả khi không có bản đồ tri thức chuyên gia.
- **Liên hệ với đồ án:** đây là đại diện của hướng "đồ thị tiên quyết suy từ dữ liệu", đối lập trực tiếp với đồ thị chuyên gia (RQ2).

**[C6] pyKT — Liu et al. (2022).** *pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models.* NeurIPS 2022 Datasets & Benchmarks.
- Có quy trình tiền xử lý chuẩn (gồm `junyi2015`) và cài đặt sẵn DKT, AKT, GKT, simpleKT, dùng làm khung đối chứng.

### 2.3. Nguồn dữ liệu
- **Junyi:** Chang, Hsu & Chen (2015), EDM 2015, và bộ dữ liệu trên PSLC DataShop. Đây là nguồn dữ liệu, không tính vào danh sách bài liên quan.
- **MOOCCubeX:** Yu et al. (2021), CIKM 2021, 4643–4652. Kho GitHub THU-KEG/MOOCCubeX có script tải dữ liệu và đã tinh chỉnh quan hệ tiên quyết cho Toán, Tâm lý học, Khoa học máy tính.

### 2.4. Tổng hợp và đề xuất

| Bài | Năm | Nơi công bố | Hạng | Loại đồ thị | Có hướng? | Đồ thị chuyên gia? |
|---|---|---|---|---|---|---|
| JKT | 2021 | Information Sciences | Q1 | bài tập–bài tập, khái niệm–khái niệm | không | không |
| Bi-CLKT | 2022 | Knowledge-Based Systems | Q1 | bài tập–bài tập (tương phản) | không | không |
| DGEKT | 2024 | ACM TOIS | Q1 | siêu đồ thị + đồ thị chuyển tiếp | có | không |
| HHSKT | 2023 | Expert Syst. Appl. | Q1 | không thuần nhất | — | không |
| AKT | 2020 | KDD | A* | không đồ thị | — | — |
| SKT | 2020 | ICDM | A* | tiên quyết + tương tự | có | một phần |
| GIKT | 2020 | ECML-PKDD | A | câu hỏi–kỹ năng | không | không |
| simpleKT | 2023 | ICLR | A* | không đồ thị | — | — |
| PSI-KT | 2024 | ICLR | A* | tiên quyết suy từ dữ liệu | có | không |

- **Khoảng trống:** chưa có đánh giá tách bạch giữa đồ thị chuyên gia và đồ thị thống kê trong **cùng một kiến trúc GKT**, kèm giao thức chống rò rỉ trên nhóm khái niệm giữ lại.
- **Đề xuất:** GKT dùng đồ thị tiên quyết chuyên gia Junyi, lan truyền hai chiều theo cách của SKT, so với GKT-Stat (cùng số cạnh), AKT, simpleKT và DKT.

---

## 3. Phương pháp giải quyết bài toán

### 3.1. Tiền xử lý
- Lọc nhật ký:
  - Bỏ học sinh có quá ít tương tác (ví dụ < 3; chọn ngưỡng và ghi lý do).
  - Sắp theo `time_done`.
  - Cắt hoặc chia chuỗi dài thành đoạn ≤ 200 bước.
- Ánh xạ `exercise_name → node_id` (0–834). Lưu bảng ánh xạ để tái lập. Khái niệm nào có trong nhật ký mà không có trong đồ thị (hoặc ngược lại) thì ghi thống kê.
- Dựng ma trận kề thưa `A` (`scipy.sparse` / `torch.sparse`) từ danh sách cạnh gốc. Báo cáo các thống kê đồ thị:
  - số cạnh, bậc vào/ra;
  - số node cô lập;
  - số thành phần liên thông;
  - có chu trình không (đồ thị tiên quyết nên là DAG).
- **`edge_type`:** nếu bộ gốc chỉ có quan hệ tiên quyết trực tiếp thì gán một loại duy nhất và ghi rõ giới hạn này. Tùy chọn: sinh thêm cạnh gián tiếp bằng bao đóng bắc cầu k bước.

### 3.2. Xử lý node cô lập (yêu cầu bắt buộc phải tự quyết định và lập luận)
So sánh ít nhất hai phương án bằng thí nghiệm ablation:
1. **Tự vòng:** `Â = A + I`, chuẩn hóa `D̂^{-1}Â` hoặc `D̂^{-1/2}ÂD̂^{-1/2}`. Node cô lập khi đó chỉ tự cập nhật, tức suy biến về DKT theo từng node.
2. **Node gốc ảo:** thêm một node nối tới mọi khái niệm nền tảng (bậc vào = 0), biểu diễn "năng lực chung".
3. **Chuẩn hóa an toàn:** quy ước `D^{-1}` với bậc 0 là 0 để tránh chia cho 0 (`NaN`).

### 3.3. Mô hình GKT tùy biến
- **Gốc:** GKT của Nakagawa et al. (2019).
  - Mỗi khái niệm có một trạng thái ẩn riêng.
  - Khi học sinh trả lời khái niệm `i`: cập nhật `i` bằng `f_self`, cập nhật hàng xóm bằng `f_neighbor`, qua cổng erase–add, rồi GRU.
  - Bài gốc chỉ thử đồ thị dense, transition, đồ thị suy từ DKT, và đồ thị học được (PAM, MHA, VAE). **Không** thử đồ thị chuyên gia.
- **Trạng thái:** `H_t ∈ R^{835×d}`.
- **Lan truyền có hướng:**
  - Tách hai chiều: ảnh hưởng xuôi (tiên quyết → khái niệm sau) và ngược (khái niệm sau → tiên quyết). Hai chiều có trọng số khác nhau: `f_fwd`, `f_bwd`.
  - Lý do: làm đúng Tích phân là bằng chứng mạnh rằng đã nắm Đạo hàm, còn chiều ngược lại yếu hơn.
- **Cập nhật:**
  - erase–add rồi GRU, chỉ cho node được trả lời và hàng xóm k bước (k = 1, 2).
  - Dùng chỉ số thưa để tránh nhân ma trận dày 835×835 ở mỗi bước.
- **Dự đoán:** `p_{t+1} = σ(W · h_{t+1}^{q_{t+1}} + b)`.
- **Hàm mất mát:** BCE trên bước kế tiếp. Tối ưu bằng Adam, dừng sớm theo AUC trên tập validation.

### 3.4. Ba biến thể đồ thị (trọng tâm RQ2)
| Biến thể | Đồ thị | Nguồn tri thức |
|---|---|---|
| GKT-Expert | cạnh tiên quyết Junyi | chuyên gia sư phạm |
| GKT-Stat | đồ thị đồng xuất hiện: P(đúng B \| đúng A), lấy ngưỡng hoặc top-k mỗi node, **chỉ tính trên tập train** | thống kê hành vi |
| GKT-Random / GKT-Dense | đồ thị ngẫu nhiên cùng số cạnh / đồ thị đầy đủ | đối chứng âm |
| AKT, DKT | không đồ thị | — |

- GKT-Stat được dựng với cùng số cạnh như GKT-Expert để so sánh công bằng.
- Tùy chọn: GKT-Expert∪Stat, hợp hai đồ thị.
- Lưu ý: từ điển dữ liệu ghi `co_occurrence_matrix` cho MOOCCubeX, nhưng để so sánh trên Junyi cần tính một ma trận tương tự từ nhật ký Junyi.

### 3.5. Pipeline mã nguồn — chạy trên Google Colab
- **Môi trường:** Google Colab với GPU T4. Mã nằm trong kho `Trilosphan/GraphK`; mỗi notebook tự `git clone` kho rồi `pip install -r requirements.txt`. Dữ liệu và kết quả lưu trên Google Drive (`/content/drive/MyDrive/GraphK/`) để không mất khi phiên Colab bị ngắt.
- **Cấu trúc kho:**
```
GraphK/
├── notebooks/
│   ├── 01_tai_du_lieu.ipynb      # tải Junyi (DataShop) + MOOCCubeX về Drive, kiểm SHA256
│   ├── 02_tien_xu_ly.ipynb       # chuỗi, node_id, A.npz, thống kê đồ thị
│   ├── 03_do_thi.ipynb           # dựng Expert / Stat (theo fold) / Random / Dense
│   ├── 04_huan_luyen.ipynb       # GKT + AKT/DKT/simpleKT (pyKT), checkpoint mỗi epoch
│   └── 05_danh_gia.ipynb         # AUC, kiểm định, bảng, hình, khái niệm giữ lại
├── src/graphk/                   # mã dùng chung: preprocess.py, graphs.py, gkt.py, train.py, eval.py
├── configs/*.yaml                # một tệp cho mỗi thí nghiệm
├── tests/                        # kiểm dựng ma trận kề, chuẩn hóa, node cô lập
└── requirements.txt
```
- **Thực hành để tái lập trên Colab:**
  - Cố định seed. Ghi phiên bản thư viện và commit Git vào tệp kết quả.
  - Lưu checkpoint lên Drive mỗi epoch. Khi Colab ngắt phiên thì chạy lại ô huấn luyện, mô hình học tiếp từ checkpoint gần nhất.
  - Mỗi ô notebook chỉ gọi hàm trong `src/graphk/`, không chứa logic dài, để tách mã khỏi notebook và để kiểm thử được.
- **Ước lượng tài nguyên:** 835 node nhân với `d = 32–64` cho ma trận trạng thái nhỏ, vừa bộ nhớ GPU T4 (16 GB). Nút thắt là độ dài chuỗi, nên cắt đoạn ≤ 200 bước và dùng phép nhân ma trận thưa.
- **Đối chứng:** AKT, DKT, simpleKT chạy bằng `pykt-toolkit` trên cùng các fold.

---

## 4. Thử nghiệm

### 4.1. Dữ liệu
- **Lưu ý số liệu:** một số khảo sát mô tả Junyi 2015 là 722 câu hỏi / 41 KC, còn đề bài ghi 835 khái niệm (bảng bài tập). Cần đếm lại trên tệp tải về và ghi rõ trong báo cáo đã dùng tập nào.
- **Junyi:** thống kê sau tiền xử lý gồm số học sinh, số tương tác, độ dài chuỗi trung bình, tỉ lệ đúng, thống kê đồ thị (mục 3.1).
- **Chia dữ liệu:** 5-fold **theo học sinh** (train / val / test). Không để học sinh nào xuất hiện ở cả train và test.
- **Giao thức chống rò rỉ (RQ3):**
  1. Dựng đồ thị thống kê **chỉ** từ học sinh của tập train trong từng fold.
  2. **Giữ lại khái niệm:** chọn khoảng 15–20% khái niệm làm tập giữ lại. Các cạnh thống kê của chúng không được dựng từ bất kỳ dữ liệu nào. Đánh giá riêng AUC trên các tương tác thuộc những khái niệm này.
  3. (Tùy chọn) chia theo thời gian: dựng đồ thị từ giai đoạn đầu, đánh giá trên giai đoạn sau.
- **MOOCCubeX (RQ4):** ánh xạ khái niệm, dựng đồ thị, chạy zero-shot hoặc fine-tune nhẹ. Cần kiểm tra trong bản tải về xem có quan hệ tiên quyết hay chỉ dựng được đồ thị đồng xuất hiện.

### 4.2. Độ đo
- **Chính:** AUC-ROC. **Phụ:** Accuracy, RMSE.
- Báo cáo trung bình ± độ lệch chuẩn qua 5 fold × nhiều seed. Dùng kiểm định cặp (paired t-test hoặc Wilcoxon) giữa các biến thể.
- **Độ đo tách bạch:**
  - ΔAUC (Expert − Stat), đo trên toàn tập và trên khái niệm giữ lại;
  - AUC theo nhóm node (cô lập / bậc thấp / bậc cao);
  - thời gian huấn luyện và bộ nhớ.

### 4.3. Kết quả
- **Bảng 1:** AUC/ACC của DKT, AKT, GKT-Dense, GKT-Random, GKT-Stat, GKT-Expert.
- **Bảng 2:** ablation xử lý node cô lập và lan truyền một chiều so với hai chiều.
- **Bảng 3:** AUC trên khái niệm giữ lại (RQ3).
- **Hình:** đường cong ROC; trực quan hóa trạng thái nắm vững trên đồ thị cho một học sinh mẫu; độ trùng cạnh giữa đồ thị Expert và Stat (Jaccard).

### 4.4. Đối sánh và thảo luận (trả lời câu hỏi phản biện)
- So với AKT: đồ thị chuyên gia có bù được khoảng cách với mô hình chú ý mạnh không?
- **Nếu GKT-Stat > GKT-Expert:**
  - Điều này không tự động phủ nhận tri thức sư phạm.
  - Đồ thị thống kê có thể chỉ khớp với chính hành vi mà tập đánh giá đo, một dạng rò rỉ gián tiếp.
  - Kiểm chứng bằng kết quả trên khái niệm giữ lại: nếu khoảng cách thu hẹp hoặc đảo chiều thì giả thuyết rò rỉ được ủng hộ.
- **Tính khái quát:** kết quả chuyển sang MOOCCubeX.
- **Giới hạn:**
  - chỉ có một loại cạnh;
  - đồ thị Junyi là đồ thị bài tập, không hẳn là đồ thị khái niệm;
  - quy mô 835 node.

---

## 5. Kết luận và hướng phát triển
- Tóm tắt câu trả lời cho RQ1–RQ4.
- Hướng phát triển:
  - học đồ thị có ràng buộc tiên nghiệm từ chuyên gia;
  - đồ thị nhiều loại cạnh (theo SKT);
  - kết hợp chú ý theo kiểu AKT với lan truyền GNN.

## Tài liệu tham khảo

### Nghiên cứu liên quan (từ 2020)
1. Song, X., Li, J., Tang, Y., Zhao, T., Chen, Y., & Guan, Z. (2021). JKT: A joint graph convolutional network based Deep Knowledge Tracing. *Information Sciences*, 580, 510–523. https://doi.org/10.1016/j.ins.2021.08.100
2. Song, X., Li, J., Lei, Q., Zhao, W., Chen, Y., & Mian, A. (2022). Bi-CLKT: Bi-Graph Contrastive Learning based Knowledge Tracing. *Knowledge-Based Systems*, 241, 108274. https://doi.org/10.1016/j.knosys.2022.108274
3. Cui, C., Yao, Y., Zhang, C., Ma, H., Ma, Y., Ren, Z., Zhang, C., & Ko, J. (2024). DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing. *ACM Transactions on Information Systems*, 42(3), 78:1–78:24. https://doi.org/10.1145/3638350
4. Ni, Q., et al. (2023). HHSKT: A learner–question interactions based heterogeneous graph neural network model for knowledge tracing. *Expert Systems with Applications*, 215, 119334. https://doi.org/10.1016/j.eswa.2022.119334
5. Abdelrahman, G., Wang, Q., & Nunes, B. P. (2023). Knowledge Tracing: A Survey. *ACM Computing Surveys*, 55(11), 1–37. https://doi.org/10.1145/3569576
6. Shen, S., Liu, Q., Huang, Z., Zheng, Y., Yin, M., Wang, M., & Chen, E. (2024). A Survey of Knowledge Tracing: Models, Variants, and Applications. *IEEE Transactions on Learning Technologies*, 17, 1898–1919. https://doi.org/10.1109/TLT.2024.3383325
7. Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. *KDD 2020*, 2330–2339. https://doi.org/10.1145/3394486.3403282
8. Tong, S., Liu, Q., Huang, W., Huang, Z., Chen, E., Liu, C., Ma, H., & Wang, S. (2020). Structure-based Knowledge Tracing: An Influence Propagation View. *IEEE ICDM 2020*, 541–550.
9. Yang, Y., Shen, J., Qu, Y., Liu, Y., Wang, K., Zhu, Y., Zhang, W., & Yu, Y. (2020). GIKT: A Graph-based Interaction Model for Knowledge Tracing. *ECML-PKDD 2020*, LNCS 12457, 299–315. https://doi.org/10.1007/978-3-030-67658-2_18
10. Liu, Z., Liu, Q., Chen, J., Huang, S., & Luo, W. (2023). simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing. *ICLR 2023*.
11. Zhou, H., Bamler, R., Wu, C. M., & Tejero-Cantero, Á. (2024). Predictive, scalable and interpretable knowledge tracing on structured domains. *ICLR 2024*.
12. Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models. *NeurIPS 2022 Datasets and Benchmarks*.

### Dữ liệu
13. Chang, H.-S., Hsu, H.-J., & Chen, K.-T. (2015). Modeling Exercise Relationships in E-Learning: A Unified Approach. *EDM 2015*.
14. Yu, J., et al. (2021). MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs. *CIKM 2021*, 4643–4652.

### Nền tảng (trước 2020, chỉ dẫn ở mục 1 và 3)
15. Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *UMUAI*, 4(4), 253–278.
16. Piech, C., et al. (2015). Deep Knowledge Tracing. *NeurIPS 2015*.
17. Nakagawa, H., Iwasawa, Y., & Matsuo, Y. (2019). Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network. *IEEE/WIC/ACM WI 2019*.
18. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.
