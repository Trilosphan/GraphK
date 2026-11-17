# Đồ án 8 — Truy vết tri thức dựa trên đồ thị (GKT) với đồ thị tiên quyết tường minh

> Môn học: Học máy (Machine Learning)
> Mô hình trọng tâm: GKT (Graph-based Knowledge Tracing), lan truyền trạng thái tri thức trên đồ thị bằng mạng nơ-ron đồ thị (Graph Neural Network, GNN)
> Dữ liệu chính: Junyi Academy (835 khái niệm, đồ thị tiên quyết do chuyên gia sư phạm xác nhận) · Dữ liệu bổ sung: MOOCCubeX
> Mức độ: Chuyên sâu · Môi trường thực nghiệm: Google Colab (GPU T4)

**Quy ước thuật ngữ:** thuật ngữ được dịch theo cách dùng trong tài liệu học máy tiếng Việt. Lần xuất hiện đầu tiên ghi kèm thuật ngữ tiếng Anh trong ngoặc. Tên mô hình, tên độ đo và tên thư viện giữ nguyên. Bảng đối chiếu thuật ngữ ở cuối văn bản.

---

## 1. Giới thiệu

### 1.1. Trí tuệ nhân tạo trong giáo dục (Artificial Intelligence in Education, AIED)
- **Phạm vi AIED:** hệ thống gia sư thông minh (Intelligent Tutoring System, ITS), học tập thích ứng (adaptive learning) và mô hình hóa người học (learner modeling).
- **Vai trò của mô hình người học:** ước lượng mức độ thành thạo (mastery) của người học đối với từng khái niệm. Ước lượng này là cơ sở để đề xuất bài tập kế tiếp và lộ trình học tập.
- **Bối cảnh nghiệp vụ:**
  - Junyi Academy có sẵn đồ thị quan hệ tiên quyết giữa các khái niệm, đã được đội ngũ sư phạm xác nhận thủ công (ví dụ: Đạo hàm là tiên quyết của Tích phân).
  - Các mô hình DKT và DKVMN ở đồ án 3–6 không khai thác thông tin cấu trúc này.

### 1.2. Truy vết tri thức (Knowledge Tracing, KT)
- **Định nghĩa:** cho chuỗi tương tác `(q_1, a_1), …, (q_t, a_t)` của một người học, trong đó `q` là câu hỏi và `a ∈ {0, 1}` là kết quả trả lời. Bài toán ước lượng trạng thái tri thức (knowledge state) của người học và dự đoán xác suất trả lời đúng `P(a_{t+1} = 1 | q_{t+1}, lịch sử)`.
- **Các họ mô hình:**
  - mô hình xác suất: BKT;
  - mô hình học sâu tuần tự: DKT, DKVMN;
  - mô hình dựa trên cơ chế chú ý (attention): SAKT, AKT;
  - mô hình dựa trên đồ thị: GKT, GIKT, SKT.
- **Hạn chế của mô hình tuần tự:** các khái niệm được xem là độc lập, hoặc chỉ phụ thuộc nhau một cách ẩn trong không gian biểu diễn. Quan hệ tiên quyết tường minh không được sử dụng.

### 1.3. Phát biểu bài toán
- **Đầu vào:**
  - Nhật ký tương tác Junyi gồm `user_id`, `exercise_name`, `correct`, `time_done`. Mỗi `exercise_name` được ánh xạ sang chỉ số nút `node_id ∈ [0, 834]`.
  - Đồ thị tiên quyết có hướng `G = (V, E)`, với `|V| = 835` và mỗi cạnh đi từ khái niệm nguồn đến khái niệm đích. Đồ thị được biểu diễn bằng ma trận kề thưa `A ∈ {0,1}^{835×835}`.
- **Đầu ra:** xác suất trả lời đúng ở bước kế tiếp đối với từng khái niệm.
- **Câu hỏi nghiên cứu (Research Question, RQ):**
  - **RQ1:** GKT sử dụng đồ thị chuyên gia có cho hiệu năng dự đoán cao hơn các mô hình không dùng đồ thị (AKT, DKT) hay không?
  - **RQ2:** Đóng góp riêng của tri thức chuyên gia so với tri thức thống kê (đồ thị đồng xuất hiện) là bao nhiêu?
  - **RQ3:** Việc so sánh hai loại đồ thị trên cùng một tập đánh giá có gây rò rỉ thông tin gián tiếp hay không? Cần thiết kế đánh giá như thế nào để so sánh công bằng?
  - **RQ4 (nếu đủ thời gian):** Mô hình có khả năng khái quát hóa (generalization) sang MOOCCubeX hay không, theo hai thiết lập: không huấn luyện thêm (zero-shot) và tinh chỉnh nhẹ (light fine-tuning)?

### 1.4. Mục tiêu và đóng góp
- Xây dựng ma trận kề từ danh sách cạnh gốc và tùy biến công thức lan truyền của GNN cho bài toán KT, kèm cơ chế xử lý nút cô lập.
- Đánh giá tách bạch ba thiết lập: đồ thị chuyên gia, đồ thị thống kê, không dùng đồ thị.
- Phân tích nguy cơ rò rỉ thông tin gián tiếp và đề xuất giao thức đánh giá trên tập khái niệm giữ lại (held-out concepts).
- Xây dựng quy trình xử lý (pipeline) tái lập được, chạy từ dữ liệu thô đến bảng kết quả. Rubric của đồ án tăng trọng số cho phần kỹ thuật mã nguồn.

---

## 2. Các nghiên cứu liên quan

**Tiêu chí chọn bài:**
- Công bố **từ năm 2020 trở lại đây**.
- Đăng trên tạp chí xếp hạng Q1–Q4 theo Scimago (SJR), hoặc trên hội nghị hạng A/A\* theo CORE.
- Có DOI hoặc trang kỷ yếu chính thức.

Các công trình nền tảng trước 2020 (BKT 1994, DKT 2015, DKVMN 2017, GKT 2019) chỉ được dẫn ở mục 1.2 và mục 3, **không** tính vào danh sách này. Hạng Q là hạng của tạp chí theo SJR và cần đối chiếu lại trên scimagojr.com theo đúng năm công bố trước khi nộp.

Mỗi công trình được phân tích theo năm khía cạnh: **phương pháp/thuật toán · dữ liệu · ưu điểm · hạn chế · kết quả đánh giá**. Mục 2.4 tổng hợp và nêu hướng đề xuất.

### 2.1. Công trình trên tạp chí (Q1)

**[J1] JKT — Song và cộng sự (2021).** *JKT: A joint graph convolutional network based Deep Knowledge Tracing.* Information Sciences, 580, 510–523. DOI 10.1016/j.ins.2021.08.100.
- **Phương pháp:**
  - Xây dựng đồ thị quan hệ bài tập–bài tập và khái niệm–khái niệm, kết hợp với quan hệ bài tập–khái niệm.
  - Học biểu diễn bằng mạng tích chập đồ thị kết hợp (joint Graph Convolutional Network, GCN). Thành phần mô hình hóa chuỗi dựa trên DKT.
- **Dữ liệu:** bốn tập dữ liệu thực, trong đó có các tập ASSISTments.
- **Ưu điểm:** liên kết được các bài tập thuộc những khái niệm khác nhau; có khả năng diễn giải, được minh họa qua nghiên cứu tình huống.
- **Hạn chế:** đồ thị được xây dựng từ thống kê và nhãn khái niệm, không sử dụng đồ thị tiên quyết của chuyên gia; không phân biệt chiều của cạnh.

**[J2] Bi-CLKT — Song và cộng sự (2022).** *Bi-CLKT: Bi-Graph Contrastive Learning based Knowledge Tracing.* Knowledge-Based Systems, 241, 108274. DOI 10.1016/j.knosys.2022.108274.
- **Phương pháp:**
  - Học tương phản (contrastive learning) hai cấp trên đồ thị con bài tập–bài tập: cấp nút cho biểu diễn bài tập, cấp đồ thị cho biểu diễn khái niệm.
  - Tầng dự đoán là mạng hồi quy (RNN) hoặc mạng tăng cường bộ nhớ (memory-augmented network).
- **Dữ liệu:** bốn tập dữ liệu thực.
- **Ưu điểm:** biểu diễn khái niệm có tính phân biệt cao hơn khi dữ liệu thưa.
- **Hạn chế:** đồ thị được suy ra từ dữ liệu hành vi nên có nguy cơ khớp với chính phân phối của tập đánh giá; chi phí tính toán lớn.

**[J3] DGEKT — Cui và cộng sự (2024).** *DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing.* ACM Transactions on Information Systems, 42(3), Article 78. DOI 10.1145/3638350.
- **Phương pháp:**
  - Cấu trúc đồ thị kép gồm siêu đồ thị (hypergraph) bài tập–khái niệm và đồ thị có hướng biểu diễn quan hệ chuyển tiếp giữa các tương tác.
  - Hai nhánh được kết hợp bằng chưng cất tri thức trực tuyến (online knowledge distillation).
- **Dữ liệu:** ba tập dữ liệu chuẩn (benchmark), so sánh với tám mô hình KT.
- **Ưu điểm:** khai thác đồng thời quan hệ không thuần nhất và quan hệ có hướng.
- **Hạn chế:** đồ thị có hướng được xây dựng từ thống kê chuyển tiếp, chưa được đối chiếu với tri thức chuyên gia.

**[J4] HHSKT — Ni và cộng sự (2023).** *HHSKT: A learner–question interactions based heterogeneous graph neural network model for knowledge tracing.* Expert Systems with Applications, 215, 119334. DOI 10.1016/j.eswa.2022.119334.
- **Phương pháp:** mạng nơ-ron đồ thị không thuần nhất (heterogeneous GNN) trên đồ thị người học–câu hỏi–khái niệm.
- **Ghi chú:** chưa tiếp cận được toàn văn; cần đọc bản gốc trước khi phân tích ưu điểm và hạn chế.

**[J5] Bài tổng quan — Abdelrahman, Wang & Nunes (2023).** *Knowledge Tracing: A Survey.* ACM Computing Surveys, 55(11), 1–37. DOI 10.1145/3569576.
- **Sử dụng cho:** mục 1.2 (lịch sử phát triển, phân loại mô hình, các tập dữ liệu chuẩn) và việc xác định khoảng trống nghiên cứu.

**[J6] Bài tổng quan — Shen và cộng sự (2024).** *A Survey of Knowledge Tracing: Models, Variants, and Applications.* IEEE Transactions on Learning Technologies, 17, 1898–1919. DOI 10.1109/TLT.2024.3383325.
- **Nội dung:** phân loại ba nhóm mô hình KT cơ bản và các biến thể; kèm thư viện mã nguồn mở.

### 2.2. Công trình trên hội nghị hạng A\*/A

**[C1] AKT — Ghosh, Heffernan & Lan (2020).** *Context-Aware Attentive Knowledge Tracing.* KDD 2020. DOI 10.1145/3394486.3403282. **Vai trò trong đồ án: mô hình cơ sở (baseline) chính.**
- **Phương pháp:**
  - Kiến trúc Transformer với cơ chế chú ý suy giảm đơn điệu (monotonic attention) theo khoảng cách thời gian.
  - Biểu diễn nhúng (embedding) câu hỏi dựa trên mô hình Rasch.
- **Ưu điểm:** thuộc nhóm có AUC cao nhất trên nhiều tập dữ liệu chuẩn.
- **Hạn chế:** không khai thác cấu trúc tiên quyết.

**[C2] SKT — Tong và cộng sự (2020).** *Structure-based Knowledge Tracing: An Influence Propagation View.* IEEE ICDM 2020, 541–550.
- **Phương pháp:** lan truyền ảnh hưởng trên cấu trúc tri thức. Quan hệ có hướng (tiên quyết) được lan truyền từng phần (partial propagation); quan hệ vô hướng (tương tự) được lan truyền đồng bộ (synchronous propagation).
- **Ưu điểm:** gần nhất với đồ án do sử dụng quan hệ tiên quyết có hướng. Đây là căn cứ cho cơ chế lan truyền hai chiều ở mục 3.3 và cho thuộc tính `edge_type`.
- **Hạn chế:** đồ thị tiên quyết chủ yếu được suy ra từ dữ liệu; chưa tách riêng đóng góp của tri thức chuyên gia.

**[C3] GIKT — Yang và cộng sự (2020).** *GIKT: A Graph-based Interaction Model for Knowledge Tracing.* ECML-PKDD 2020, LNCS 12457, 299–315. DOI 10.1007/978-3-030-67658-2_18.
- **Phương pháp:**
  - Dùng GCN để lan truyền biểu diễn nhúng trên đồ thị câu hỏi–kỹ năng.
  - Mô hình hóa tương tác giữa trạng thái hiện tại, các bài tập liên quan trong lịch sử, câu hỏi đích và các kỹ năng liên quan.
- **Kết quả:** đạt kết quả tốt nhất hiện có (state of the art) trên ba tập dữ liệu, AUC tăng ít nhất khoảng 1 điểm phần trăm.

**[C4] simpleKT — Liu và cộng sự (2023).** *simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing.* ICLR 2023.
- **Vai trò:** mô hình cơ sở mạnh bổ sung.
- **Ý nghĩa đối với đồ án:** công trình cho thấy nhiều mô hình phức tạp không vượt được một mô hình cơ sở đơn giản khi được đánh giá theo giao thức chuẩn hóa. Đây là lý do đồ án tuân theo giao thức đánh giá của pyKT.

**[C5] PSI-KT — Zhou và cộng sự (2024).** *Predictive, scalable and interpretable knowledge tracing on structured domains.* ICLR 2024.
- **Phương pháp:** mô hình sinh phân cấp (hierarchical generative model), suy luận đồng thời đặc trưng nhận thức của từng người học và một đồ thị tiên quyết dùng chung. Độ mạnh của quan hệ tiên quyết được ước lượng trực tiếp từ dữ liệu làm bài.
- **Ưu điểm:** có khả năng diễn giải theo thiết kế; áp dụng được khi không có bản đồ tri thức của chuyên gia.
- **Liên hệ với đồ án:** đại diện cho hướng "đồ thị tiên quyết suy ra từ dữ liệu", đối lập trực tiếp với đồ thị chuyên gia trong RQ2.

**[C6] pyKT — Liu và cộng sự (2022).** *pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models.* NeurIPS 2022 Datasets & Benchmarks Track.
- **Nội dung:** quy trình tiền xử lý chuẩn hóa (có `junyi2015`) và cài đặt sẵn DKT, AKT, GKT, simpleKT. Đồ án dùng làm khung đánh giá đối chứng.

### 2.3. Nguồn dữ liệu
- **Junyi:** Chang, Hsu & Chen (2015), EDM 2015; dữ liệu công bố trên PSLC DataShop. Công trình này được trích dẫn như nguồn dữ liệu, không tính vào danh sách nghiên cứu liên quan.
- **MOOCCubeX:** Yu và cộng sự (2021), CIKM 2021, 4643–4652. Kho mã THU-KEG/MOOCCubeX cung cấp script tải dữ liệu; quan hệ tiên quyết các ngành Toán, Tâm lý học và Khoa học máy tính đã được hiệu chỉnh.

### 2.4. Tổng hợp và hướng đề xuất

| Công trình | Năm | Nơi công bố | Hạng | Loại đồ thị | Có hướng | Đồ thị chuyên gia |
|---|---|---|---|---|---|---|
| JKT | 2021 | Information Sciences | Q1 | bài tập–bài tập, khái niệm–khái niệm | không | không |
| Bi-CLKT | 2022 | Knowledge-Based Systems | Q1 | bài tập–bài tập (học tương phản) | không | không |
| DGEKT | 2024 | ACM TOIS | Q1 | siêu đồ thị + đồ thị chuyển tiếp | có | không |
| HHSKT | 2023 | Expert Syst. Appl. | Q1 | đồ thị không thuần nhất | — | không |
| AKT | 2020 | KDD | A\* | không dùng đồ thị | — | — |
| SKT | 2020 | ICDM | A\* | tiên quyết + tương tự | có | một phần |
| GIKT | 2020 | ECML-PKDD | A | câu hỏi–kỹ năng | không | không |
| simpleKT | 2023 | ICLR | A\* | không dùng đồ thị | — | — |
| PSI-KT | 2024 | ICLR | A\* | tiên quyết suy ra từ dữ liệu | có | không |

- **Khoảng trống nghiên cứu:** chưa có đánh giá tách bạch giữa đồ thị chuyên gia và đồ thị thống kê trong **cùng một kiến trúc GKT**, kèm giao thức chống rò rỉ trên tập khái niệm giữ lại.
- **Hướng đề xuất:** GKT sử dụng đồ thị tiên quyết chuyên gia của Junyi, với cơ chế lan truyền hai chiều theo SKT. Mô hình được so sánh với GKT-Stat (cùng số cạnh), AKT, simpleKT và DKT.

---

## 3. Phương pháp giải quyết bài toán

### 3.1. Tiền xử lý dữ liệu
- **Lọc và sắp xếp nhật ký:**
  - Loại người học có quá ít tương tác (ví dụ dưới 3); ngưỡng được chọn có lập luận và ghi rõ trong báo cáo.
  - Sắp xếp tương tác theo `time_done`.
  - Chia các chuỗi dài thành các đoạn không quá 200 bước.
- **Ánh xạ chỉ số:** `exercise_name → node_id` (0–834). Bảng ánh xạ được lưu lại để bảo đảm tái lập. Thống kê số khái niệm có trong nhật ký nhưng không có trong đồ thị, và ngược lại.
- **Xây dựng ma trận kề thưa `A`** (`scipy.sparse` / `torch.sparse`) từ danh sách cạnh gốc. Báo cáo các đặc trưng của đồ thị:
  - số cạnh; phân phối bậc vào và bậc ra;
  - số nút cô lập;
  - số thành phần liên thông;
  - kiểm tra chu trình (đồ thị tiên quyết cần là đồ thị có hướng không chu trình, DAG).
- **Thuộc tính `edge_type`:** nếu dữ liệu gốc chỉ có quan hệ tiên quyết trực tiếp, gán một loại cạnh duy nhất và nêu rõ giới hạn này. Phương án mở rộng: sinh cạnh gián tiếp bằng bao đóng bắc cầu (transitive closure) k bước.

### 3.2. Xử lý nút cô lập
Đề bài yêu cầu tự quyết định và lập luận cho lựa chọn này. Ít nhất hai phương án được so sánh bằng nghiên cứu cắt bỏ (ablation study):
1. **Thêm vòng tự thân (self-loop):** `Â = A + I`, chuẩn hóa theo hàng `D̂^{-1}Â` hoặc chuẩn hóa đối xứng `D̂^{-1/2}ÂD̂^{-1/2}`. Khi đó nút cô lập chỉ tự cập nhật trạng thái của chính nó, tương đương DKT trên từng nút.
2. **Thêm nút gốc ảo (virtual root node):** nút ảo nối tới mọi khái niệm nền tảng (bậc vào bằng 0), đóng vai trò biểu diễn năng lực chung của người học.
3. **Chuẩn hóa an toàn số học:** quy ước phần tử nghịch đảo của bậc 0 trong `D^{-1}` bằng 0, tránh phép chia cho 0 sinh giá trị `NaN`.

### 3.3. Mô hình GKT tùy biến
- **Mô hình gốc** (Nakagawa và cộng sự, 2019):
  - Mỗi khái niệm có một trạng thái ẩn (hidden state) riêng.
  - Khi người học trả lời khái niệm `i`: trạng thái của `i` được cập nhật bằng hàm `f_self`, trạng thái của các nút lân cận được cập nhật bằng hàm `f_neighbor`. Sau đó áp dụng cơ chế cổng xóa–thêm (erase–add gate) và đơn vị hồi quy có cổng (Gated Recurrent Unit, GRU).
  - Công trình gốc chỉ thử nghiệm đồ thị đầy đủ (dense), đồ thị chuyển tiếp (transition), đồ thị suy ra từ DKT và các đồ thị học được (PAM, MHA, VAE). Đồ thị chuyên gia **chưa** được thử nghiệm.
- **Trạng thái tri thức:** `H_t ∈ R^{835×d}`.
- **Lan truyền có hướng:**
  - Tách hai chiều lan truyền: chiều xuôi (khái niệm tiên quyết → khái niệm phụ thuộc) và chiều ngược (khái niệm phụ thuộc → khái niệm tiên quyết). Mỗi chiều có tham số riêng, `f_fwd` và `f_bwd`.
  - Lập luận: trả lời đúng bài Tích phân là bằng chứng mạnh về việc đã thành thạo Đạo hàm, trong khi chiều ngược lại cho bằng chứng yếu hơn.
- **Cập nhật trạng thái:**
  - Áp dụng cổng xóa–thêm và GRU, chỉ cho nút được trả lời cùng lân cận trong phạm vi k bước (k = 1, 2).
  - Sử dụng phép toán trên ma trận thưa, tránh nhân ma trận dày 835×835 ở mỗi bước thời gian.
- **Tầng dự đoán:** `p_{t+1} = σ(W · h_{t+1}^{q_{t+1}} + b)`.
- **Huấn luyện:**
  - Hàm mất mát entropy chéo nhị phân (binary cross-entropy) trên dự đoán ở bước kế tiếp.
  - Tối ưu bằng Adam.
  - Dừng sớm (early stopping) theo AUC trên tập kiểm định (validation set).

### 3.4. Các thiết lập đồ thị (trọng tâm của RQ2)
| Thiết lập | Đồ thị | Nguồn tri thức |
|---|---|---|
| GKT-Expert | cạnh tiên quyết của Junyi | chuyên gia sư phạm |
| GKT-Stat | đồ thị đồng xuất hiện: ước lượng P(đúng B \| đúng A), giữ cạnh theo ngưỡng hoặc top-k mỗi nút, **chỉ tính trên tập huấn luyện** | thống kê hành vi |
| GKT-Random / GKT-Dense | đồ thị ngẫu nhiên cùng số cạnh / đồ thị đầy đủ | đối chứng âm (negative control) |
| AKT, DKT | không dùng đồ thị | — |

- GKT-Stat được xây dựng với số cạnh bằng GKT-Expert để so sánh công bằng.
- Thiết lập mở rộng: GKT-Expert∪Stat, sử dụng hợp của hai đồ thị.
- **Lưu ý:** từ điển dữ liệu chỉ định nghĩa `co_occurrence_matrix` cho MOOCCubeX. Để so sánh trên Junyi, cần tính ma trận tương ứng từ nhật ký tương tác Junyi.

### 3.5. Quy trình thực nghiệm trên Google Colab
- **Môi trường thực thi:**
  - Google Colab với GPU T4. Mã nguồn đặt tại kho `Trilosphan/GraphK`; mỗi notebook tự `git clone` kho mã và cài đặt bằng `pip install -r requirements.txt`.
  - Dữ liệu và kết quả lưu trên Google Drive (`/content/drive/MyDrive/GraphK/`) để không bị mất khi phiên Colab bị ngắt.
- **Cấu trúc kho mã:**
```
GraphK/
├── notebooks/
│   ├── 01_tai_du_lieu.ipynb      # tải Junyi (DataShop) và MOOCCubeX về Drive, kiểm tra SHA256
│   ├── 02_tien_xu_ly.ipynb       # tạo chuỗi tương tác, node_id, A.npz, đặc trưng đồ thị
│   ├── 03_do_thi.ipynb           # xây dựng đồ thị Expert / Stat (theo từng phần) / Random / Dense
│   ├── 04_huan_luyen.ipynb       # huấn luyện GKT và AKT/DKT/simpleKT (pyKT), lưu điểm kiểm tra mỗi epoch
│   └── 05_danh_gia.ipynb         # AUC, kiểm định thống kê, bảng, hình, tập khái niệm giữ lại
├── src/graphk/                   # mô-đun dùng chung: preprocess.py, graphs.py, gkt.py, train.py, eval.py
├── configs/*.yaml                # mỗi thực nghiệm một tệp cấu hình
├── tests/                        # kiểm thử xây dựng ma trận kề, chuẩn hóa, xử lý nút cô lập
└── requirements.txt
```
- **Bảo đảm tính tái lập (reproducibility):**
  - Cố định hạt giống ngẫu nhiên (random seed). Ghi phiên bản thư viện và mã commit Git vào tệp kết quả.
  - Lưu điểm kiểm tra (checkpoint) lên Drive sau mỗi epoch. Khi phiên Colab bị ngắt, chạy lại ô huấn luyện để tiếp tục từ điểm kiểm tra gần nhất.
  - Mỗi ô notebook chỉ gọi hàm trong `src/graphk/`, không chứa logic dài, nhằm tách mã khỏi notebook và để mã kiểm thử được.
- **Ước lượng tài nguyên:**
  - Ma trận trạng thái 835 × `d` (với `d` = 32–64) có kích thước nhỏ, phù hợp bộ nhớ GPU T4 (16 GB).
  - Nút thắt tính toán là độ dài chuỗi, nên giới hạn đoạn chuỗi không quá 200 bước và dùng phép nhân ma trận thưa.
- **Mô hình cơ sở:** AKT, DKT, simpleKT được huấn luyện bằng `pykt-toolkit` trên cùng các phần chia dữ liệu.

---

## 4. Thực nghiệm

### 4.1. Dữ liệu
- **Lưu ý về số liệu:** một số bài tổng quan mô tả Junyi 2015 gồm 722 câu hỏi và 41 thành phần tri thức (knowledge component, KC), trong khi đề bài ghi 835 khái niệm (bảng bài tập). Cần thống kê lại trên tệp tải về và ghi rõ trong báo cáo tập nào được sử dụng.
- **Thống kê mô tả Junyi sau tiền xử lý:** số người học, số tương tác, độ dài chuỗi trung bình, tỉ lệ trả lời đúng, các đặc trưng đồ thị (mục 3.1).
- **Phân chia dữ liệu:** kiểm định chéo 5 phần (5-fold cross-validation) **theo người học**, mỗi phần gồm tập huấn luyện, tập kiểm định và tập kiểm tra (test set). Không người học nào xuất hiện đồng thời trong tập huấn luyện và tập kiểm tra.
- **Giao thức chống rò rỉ dữ liệu (data leakage) cho RQ3:**
  1. Đồ thị thống kê của mỗi phần **chỉ** được xây dựng từ người học thuộc tập huấn luyện của phần đó.
  2. **Tập khái niệm giữ lại:** chọn ngẫu nhiên khoảng 15–20% khái niệm. Các cạnh thống kê của những khái niệm này không được xây dựng từ bất kỳ dữ liệu nào. AUC được đánh giá riêng trên các tương tác thuộc tập khái niệm này.
  3. Phương án mở rộng: phân chia theo thời gian. Đồ thị được xây dựng từ giai đoạn đầu và đánh giá trên giai đoạn sau.
- **MOOCCubeX (RQ4):** ánh xạ khái niệm, xây dựng đồ thị, đánh giá theo thiết lập zero-shot và tinh chỉnh nhẹ. Cần kiểm tra bản tải về có chứa quan hệ tiên quyết hay chỉ xây dựng được đồ thị đồng xuất hiện.

### 4.2. Độ đo đánh giá
- **Độ đo chính:** diện tích dưới đường cong ROC (AUC-ROC).
- **Độ đo phụ:** độ chính xác (accuracy, ACC); căn bậc hai của sai số toàn phương trung bình (RMSE).
- **Báo cáo thống kê:**
  - Trung bình ± độ lệch chuẩn qua 5 phần × nhiều hạt giống ngẫu nhiên.
  - Kiểm định thống kê ghép cặp (paired t-test hoặc kiểm định Wilcoxon) giữa các thiết lập.
- **Độ đo phân tích tách bạch:**
  - ΔAUC (Expert − Stat), tính trên toàn tập kiểm tra và trên tập khái niệm giữ lại;
  - AUC theo nhóm nút: nút cô lập, nút bậc thấp, nút bậc cao;
  - thời gian huấn luyện và bộ nhớ sử dụng.

### 4.3. Kết quả
- **Bảng 1:** AUC và ACC của DKT, AKT, simpleKT, GKT-Dense, GKT-Random, GKT-Stat, GKT-Expert.
- **Bảng 2:** nghiên cứu cắt bỏ về cách xử lý nút cô lập và lan truyền một chiều so với hai chiều.
- **Bảng 3:** AUC trên tập khái niệm giữ lại (RQ3).
- **Hình:**
  - đường cong ROC;
  - trực quan hóa trạng thái thành thạo trên đồ thị của một người học điển hình;
  - mức trùng lặp cạnh giữa đồ thị Expert và đồ thị Stat (hệ số Jaccard).

### 4.4. Đối sánh và thảo luận (trả lời các câu hỏi phản biện)
- **So với AKT:** đồ thị chuyên gia có bù được khoảng cách hiệu năng so với một mô hình chú ý mạnh hay không?
- **Trường hợp GKT-Stat đạt AUC cao hơn GKT-Expert:**
  - Kết quả này không đủ để phủ nhận giá trị của tri thức sư phạm.
  - Đồ thị thống kê có thể chỉ khớp tốt hơn với chính phân phối hành vi mà tập đánh giá đo lường. Đây là một dạng rò rỉ thông tin gián tiếp.
  - Kiểm chứng bằng kết quả trên tập khái niệm giữ lại: nếu khoảng cách thu hẹp hoặc đảo chiều thì giả thuyết rò rỉ được ủng hộ.
- **Khả năng khái quát hóa:** kết quả chuyển giao sang MOOCCubeX.
- **Giới hạn của nghiên cứu:**
  - chỉ có một loại cạnh;
  - đồ thị Junyi là đồ thị bài tập, chưa hẳn là đồ thị khái niệm;
  - quy mô 835 nút.

---

## 5. Kết luận và hướng phát triển
- Tổng kết câu trả lời cho RQ1–RQ4.
- **Hướng phát triển:**
  - học cấu trúc đồ thị với tri thức chuyên gia làm phân phối tiên nghiệm (prior);
  - đồ thị nhiều loại cạnh, theo hướng của SKT;
  - kết hợp cơ chế chú ý của AKT với lan truyền trên GNN.

---

## Bảng đối chiếu thuật ngữ

| Tiếng Anh | Tiếng Việt |
|---|---|
| Knowledge Tracing (KT) | truy vết tri thức |
| knowledge state / mastery | trạng thái tri thức / mức độ thành thạo |
| Graph Neural Network (GNN) | mạng nơ-ron đồ thị |
| Graph Convolutional Network (GCN) | mạng tích chập đồ thị |
| adjacency matrix / degree matrix | ma trận kề / ma trận bậc |
| node / edge / isolated node | nút / cạnh / nút cô lập |
| self-loop / virtual node | vòng tự thân / nút ảo |
| directed acyclic graph (DAG) | đồ thị có hướng không chu trình |
| transitive closure | bao đóng bắc cầu |
| message passing / propagation | truyền thông điệp / lan truyền |
| hidden state / gate / GRU | trạng thái ẩn / cổng / đơn vị hồi quy có cổng |
| attention mechanism | cơ chế chú ý |
| embedding | biểu diễn nhúng |
| contrastive learning | học tương phản |
| knowledge distillation | chưng cất tri thức |
| hypergraph / heterogeneous graph | siêu đồ thị / đồ thị không thuần nhất |
| baseline | mô hình cơ sở |
| ablation study | nghiên cứu cắt bỏ |
| training / validation / test set | tập huấn luyện / tập kiểm định / tập kiểm tra |
| k-fold cross-validation | kiểm định chéo k phần |
| data leakage | rò rỉ dữ liệu |
| held-out concepts | tập khái niệm giữ lại |
| early stopping / checkpoint | dừng sớm / điểm kiểm tra |
| binary cross-entropy | entropy chéo nhị phân |
| random seed | hạt giống ngẫu nhiên |
| generalization / fine-tuning / zero-shot | khái quát hóa / tinh chỉnh / không huấn luyện thêm |
| paired test | kiểm định ghép cặp |
| state of the art (SOTA) | kết quả tốt nhất hiện có |
| reproducibility | tính tái lập |

---

## Tài liệu tham khảo

### Nghiên cứu liên quan (từ năm 2020)
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
12. Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models. *NeurIPS 2022 Datasets and Benchmarks Track*.

### Nguồn dữ liệu
13. Chang, H.-S., Hsu, H.-J., & Chen, K.-T. (2015). Modeling Exercise Relationships in E-Learning: A Unified Approach. *EDM 2015*.
14. Yu, J., et al. (2021). MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs. *CIKM 2021*, 4643–4652.

### Công trình nền tảng (trước 2020, dẫn ở mục 1 và mục 3)
15. Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.
16. Piech, C., et al. (2015). Deep Knowledge Tracing. *NeurIPS 2015*.
17. Nakagawa, H., Iwasawa, Y., & Matsuo, Y. (2019). Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network. *IEEE/WIC/ACM WI 2019*.
18. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.
