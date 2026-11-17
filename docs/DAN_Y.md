# Đồ án 8 — Truy vết tri thức dựa trên đồ thị (GKT) với đồ thị tiên quyết tường minh

> Môn học: Học máy (Machine Learning)
> Mô hình trọng tâm: GKT (Graph-based Knowledge Tracing), lan truyền trạng thái tri thức trên đồ thị bằng mạng nơ-ron đồ thị (Graph Neural Network, GNN)
> Dữ liệu chính: Junyi Academy (835 khái niệm, đồ thị tiên quyết do chuyên gia sư phạm xác nhận) · Dữ liệu bổ sung: MOOCCubeX
> Mức độ: Chuyên sâu · Môi trường thực nghiệm: Google Colab (GPU T4)

**Quy ước thuật ngữ:** thuật ngữ được dịch theo cách dùng trong tài liệu học máy tiếng Việt. Lần xuất hiện đầu tiên ghi kèm thuật ngữ tiếng Anh trong ngoặc. Tên mô hình, tên độ đo và tên thư viện giữ nguyên. Bảng đối chiếu thuật ngữ ở Phụ lục E.

---

## Tóm tắt

- **Bối cảnh và vấn đề:** các mô hình truy vết tri thức tuần tự (DKT, AKT) không dùng đồ thị tiên quyết mà đội ngũ sư phạm của Junyi Academy đã xác nhận. Các mô hình dựa trên đồ thị hiện có lại dùng đồ thị suy ra từ dữ liệu, nên chưa ai tách riêng được giá trị của tri thức chuyên gia.
- **Phương pháp:** GKT lan truyền hai chiều trên đồ thị tiên quyết có hướng: hàm thông điệp riêng cho chiều xuôi (tiên quyết → phụ thuộc) và chiều ngược, cài đặt bằng danh sách cạnh nén. Cùng một kiến trúc được huấn luyện trên ba đồ thị có cùng 979 cạnh: chuyên gia, thống kê (giá trị gia tăng, chỉ tính trên tập huấn luyện) và ngẫu nhiên.
- **Thiết lập:** Junyi 2015 với 835 khái niệm, 191.874 người học và 25,85 triệu tương tác. Dữ liệu chia theo người học; tập kiểm tra gồm 38.375 người học. Đồ án dùng 108 khái niệm giữ lại để kiểm soát rò rỉ gián tiếp. Mô hình cơ sở là DKT và AKT. Độ đo là AUC, ACC, RMSE, kèm kiểm định DeLong ghép cặp.
- **Kết quả chính:** ⟨điền từ notebook 05: AUC của GKT-Expert so với GKT-Stat, GKT-Random, DKT, AKT; chênh lệch trên khái niệm giữ lại⟩.
- **Phát hiện về dữ liệu:**
  - đồ thị chuyên gia có một chu trình 3 nút và hai vòng tự thân;
  - đồ thị thống kê chỉ trùng khoảng 1% cạnh với đồ thị chuyên gia;
  - tương quan hành vi không phân biệt được chiều tiên quyết.
- **Kết luận:** ⟨điền sau khi có kết quả⟩.

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

Các công trình nền tảng trước 2020 (BKT 1994, DKT 2015, DKVMN 2017, GKT 2019) chỉ được dẫn ở mục 1.2 và mục 3. Hạng Q là hạng của tạp chí theo SJR, cần đối chiếu lại trên scimagojr.com theo năm công bố trước khi nộp.

**Cách tổ chức:** các công trình được chia thành ba nhóm phương pháp cạnh tranh hoặc đi trước GKT-Expert. Mỗi nhóm trình bày theo bốn ý:
- (i) các công trình tiêu biểu, mỗi công trình gồm phương pháp, dữ liệu, kết quả;
- (ii) giả thuyết về hạn chế chung của nhóm;
- (iii) bằng chứng cho hạn chế đó;
- (iv) cầu nối, tức GKT-Expert khắc phục hạn chế đó như thế nào và đồ án kiểm chứng bằng thực nghiệm nào.

Hai bài tổng quan dùng cho bối cảnh chung ở mục 1:
- **[J5]** Abdelrahman, Wang & Nunes (2023), *Knowledge Tracing: A Survey*, ACM Computing Surveys 55(11), Q1;
- **[J6]** Shen và cộng sự (2024), *A Survey of Knowledge Tracing: Models, Variants, and Applications*, IEEE Transactions on Learning Technologies 17, Q1.

### 2.1. Nhóm 1 — Truy vết tri thức không dùng đồ thị

**Công trình tiêu biểu:**
- **[C1] AKT** — Ghosh, Heffernan & Lan, KDD 2020 (A\*). *Vai trò trong đồ án: mô hình cơ sở chính.*
  - Phương pháp: Transformer với chú ý suy giảm đơn điệu theo khoảng cách thời gian, biểu diễn nhúng câu hỏi theo mô hình Rasch.
  - Kết quả: thuộc nhóm có AUC cao nhất trên nhiều tập dữ liệu chuẩn.
- **[C4] simpleKT** — Liu và cộng sự, ICLR 2023 (A\*).
  - Phương pháp: chú ý tích vô hướng thông thường kết hợp biến thiên theo câu hỏi (Rasch).
  - Kết quả: luôn nằm trong top 3 AUC trên 7 tập dữ liệu, so với 12 mô hình học sâu.
- **[C6] pyKT** — Liu và cộng sự, NeurIPS 2022 Datasets & Benchmarks.
  - Nội dung: quy trình tiền xử lý và đánh giá chuẩn hóa cho truy vết tri thức học sâu (có `junyi2015`), cùng các cài đặt sẵn DKT, AKT, GKT.

**Giả thuyết về hạn chế:** quan hệ giữa các khái niệm chỉ được học ngầm qua biểu diễn nhúng và trọng số chú ý. Khi một khái niệm có ít dữ liệu, hoặc người học chưa từng làm khái niệm đó, mô hình không có kênh tường minh để chuyển bằng chứng từ các khái niệm tiên quyết hay phụ thuộc.

**Bằng chứng:**
- simpleKT cho thấy phần lớn cải tiến kiến trúc chỉ cho chênh lệch AUC rất nhỏ khi đánh giá chuẩn hóa: biểu diễn ngầm đã bão hòa.
- AKT và simpleKT đều không có thành phần nào mã hóa quan hệ tiên quyết.

**Cầu nối sang GKT-Expert:**
- GKT-Expert đưa đồ thị tiên quyết vào làm kênh lan truyền tường minh: trả lời khái niệm `i` cập nhật trực tiếp trạng thái của các khái niệm tiên quyết và phụ thuộc của `i`.
- Kiểm chứng (RQ1): so AUC của GKT-Expert với DKT và AKT trên cùng tập kiểm tra, cùng giao thức pyKT, kèm kiểm định DeLong ghép cặp.

### 2.2. Nhóm 2 — Truy vết tri thức dùng đồ thị suy ra từ dữ liệu

**Công trình tiêu biểu:**
- **[J1] JKT** — Song và cộng sự, *Information Sciences* 580 (2021), Q1.
  - Phương pháp: GCN kết hợp trên đồ thị bài tập–bài tập và khái niệm–khái niệm, phần chuỗi theo DKT.
  - Dữ liệu và kết quả: 4 tập dữ liệu, vượt các mô hình cơ sở, có nghiên cứu tình huống về khả năng diễn giải.
- **[J2] Bi-CLKT** — Song và cộng sự, *Knowledge-Based Systems* 241 (2022), Q1.
  - Phương pháp: học tương phản hai cấp (nút, đồ thị) trên đồ thị con bài tập–bài tập.
  - Dữ liệu và kết quả: 4 tập dữ liệu, vượt các mô hình cơ sở.
- **[J3] DGEKT** — Cui và cộng sự, *ACM Transactions on Information Systems* 42(3) (2024), Q1.
  - Phương pháp: siêu đồ thị bài tập–khái niệm và đồ thị chuyển tiếp có hướng, kết hợp bằng chưng cất tri thức trực tuyến.
  - Dữ liệu và kết quả: 3 tập dữ liệu, so với 8 mô hình cơ sở.
- **[J4] HHSKT** — Ni và cộng sự, *Expert Systems with Applications* 215 (2023), Q1.
  - Phương pháp: GNN không thuần nhất trên đồ thị người học–câu hỏi–khái niệm. *Cần đọc toàn văn để bổ sung chi tiết.*
- **[C3] GIKT** — Yang và cộng sự, ECML-PKDD 2020 (A), LNCS 12457.
  - Phương pháp: GCN lan truyền biểu diễn nhúng trên đồ thị câu hỏi–kỹ năng.
  - Kết quả: AUC tăng ít nhất khoảng 1 điểm phần trăm trên 3 tập dữ liệu.

**Giả thuyết về hạn chế:** đồ thị được suy ra từ chính dữ liệu hành vi, nên có thể chỉ phản ánh tương quan (ví dụ năng lực chung của người học) chứ không phản ánh quan hệ tiên quyết. Khi đánh giá trên cùng phân phối hành vi, AUC cao có thể là rò rỉ thông tin gián tiếp. Ngoài ra đồ thị không phủ được các khái niệm thiếu dữ liệu.

**Bằng chứng:**
- Các công trình trên chỉ đánh giá trên toàn tập kiểm tra, không đánh giá riêng trên những khái niệm chưa dùng để xây dựng đồ thị.
- Phân tích của đồ án (notebook 03) cho thấy đồ thị thống kê trên Junyi chỉ trùng khoảng 1% cạnh với đồ thị chuyên gia.
- Trung vị giá trị gia tăng theo chiều xuôi (0,020) và chiều ngược (0,022) gần như bằng nhau: tương quan hành vi không phân biệt được chiều tiên quyết.

**Cầu nối sang GKT-Expert:**
- Đồ thị chuyên gia độc lập với dữ liệu làm bài, nên vẫn có cạnh cho khái niệm ít dữ liệu và có chiều xác định.
- Kiểm chứng (RQ2, RQ3):
  - cùng kiến trúc GKT, cùng 979 cạnh, so đồ thị chuyên gia với đồ thị thống kê (chỉ tính trên tập huấn luyện) và đồ thị ngẫu nhiên;
  - đo AUC riêng trên 108 khái niệm giữ lại, nơi đồ thị thống kê không có cạnh nào.

### 2.3. Nhóm 3 — Truy vết tri thức có quan hệ tiên quyết

**Công trình tiêu biểu:**
- **[C2] SKT** — Tong và cộng sự, IEEE ICDM 2020 (A\*).
  - Phương pháp: lan truyền ảnh hưởng trên cấu trúc tri thức; quan hệ tiên quyết (có hướng) lan truyền từng phần, quan hệ tương tự (vô hướng) lan truyền đồng bộ.
- **[C5] PSI-KT** — Zhou và cộng sự, ICLR 2024 (A\*).
  - Phương pháp: mô hình sinh phân cấp suy ra đồng thời đặc trưng nhận thức của từng người học và một đồ thị tiên quyết dùng chung.
  - Kết quả: dự đoán nhiều bước tốt, suy luận mở rộng được, có khả năng diễn giải; chỉ đánh giá trên dữ liệu môn Toán.

**Giả thuyết về hạn chế:** cả hai công trình đều ước lượng cấu trúc tiên quyết từ dữ liệu, hoặc dùng cấu trúc mà không tách riêng nguồn gốc của nó. Vì vậy chưa trả lời được câu hỏi: tri thức chuyên gia đóng góp bao nhiêu so với việc chỉ có thêm một cấu trúc bất kỳ.

**Bằng chứng:**
- Trong GKT gốc (Nakagawa và cộng sự, 2019), các đồ thị thử nghiệm đều là đồ thị đầy đủ, đồ thị chuyển tiếp, hoặc đồ thị học được (PAM, MHA, VAE); không có đồ thị chuyên gia.
- PSI-KT nêu rõ ưu điểm là áp dụng được khi không có bản đồ tri thức chuyên gia, tức không so sánh với bản đồ đó.

**Cầu nối sang GKT-Expert:**
- GKT-Expert dùng trực tiếp đồ thị chuyên gia của Junyi, với cơ chế lan truyền hai chiều theo tinh thần của SKT (hàm thông điệp riêng cho chiều xuôi và chiều ngược).
- Kiểm chứng:
  - đối chứng âm GKT-Random (cùng số cạnh) tách hiệu ứng "có cấu trúc" khỏi hiệu ứng "đúng cấu trúc";
  - nghiên cứu cắt bỏ một chiều/hai chiều kiểm tra giá trị của thông tin về chiều cạnh.

### 2.4. Tổng hợp

| Nhóm | Công trình | Năm | Nơi công bố | Hạng | Loại đồ thị | Có hướng | Đồ thị chuyên gia |
|---|---|---|---|---|---|---|---|
| 1 | AKT | 2020 | KDD | A\* | không dùng | — | — |
| 1 | simpleKT | 2023 | ICLR | A\* | không dùng | — | — |
| 2 | JKT | 2021 | Information Sciences | Q1 | bài tập–bài tập, khái niệm–khái niệm | không | không |
| 2 | Bi-CLKT | 2022 | Knowledge-Based Systems | Q1 | bài tập–bài tập (học tương phản) | không | không |
| 2 | DGEKT | 2024 | ACM TOIS | Q1 | siêu đồ thị + đồ thị chuyển tiếp | có | không |
| 2 | HHSKT | 2023 | Expert Syst. Appl. | Q1 | không thuần nhất | — | không |
| 2 | GIKT | 2020 | ECML-PKDD | A | câu hỏi–kỹ năng | không | không |
| 3 | SKT | 2020 | ICDM | A\* | tiên quyết + tương tự | có | một phần |
| 3 | PSI-KT | 2024 | ICLR | A\* | tiên quyết suy ra từ dữ liệu | có | không |
| — | **GKT-Expert (đồ án)** | 2026 | — | — | **tiên quyết chuyên gia, lan truyền hai chiều** | **có** | **có** |

**Khoảng trống nghiên cứu:** chưa có đánh giá tách bạch giữa đồ thị chuyên gia và đồ thị thống kê trong cùng một kiến trúc GKT, cùng số cạnh, kèm giao thức chống rò rỉ trên tập khái niệm giữ lại.

**Nguồn dữ liệu** (trích dẫn như nguồn dữ liệu, không tính vào danh sách trên):
- Junyi: Chang, Hsu & Chen (2015), EDM 2015; PSLC DataShop.
- MOOCCubeX: Yu và cộng sự (2021), CIKM 2021, 4643–4652.

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
| GKT-Stat | đồ thị đồng xuất hiện: điểm giá trị gia tăng AV(A → B) = P(thành thạo B \| thành thạo A) − P(thành thạo B), mỗi nút ưu tiên một cạnh vào, phần còn lại lấy theo điểm toàn cục, **chỉ tính trên tập huấn luyện** | thống kê hành vi |
| GKT-Random / GKT-Dense | đồ thị ngẫu nhiên cùng số cạnh / đồ thị đầy đủ | đối chứng âm (negative control) |
| AKT, DKT | không dùng đồ thị | — |

- GKT-Stat được xây dựng với số cạnh bằng GKT-Expert để so sánh công bằng.
- **Kết quả xây dựng đồ thị** (notebook 03, ngày 17/11/2026):
  - Đồ thị thống kê trùng với đồ thị chuyên gia khoảng 1% cạnh (Jaccard có hướng ≈ 0,01).
  - Trung vị AV của cạnh chuyên gia theo chiều xuôi (0,020) và chiều ngược (0,022) gần như bằng nhau: dữ liệu hành vi không phân biệt được chiều tiên quyết.
  - Đồ thị thống kê có khoảng 330 nút cô lập, so với 59 của đồ thị chuyên gia. Trong đó 221 nút buộc phải cô lập: 108 khái niệm giữ lại và 113 khái niệm không có tương tác.
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
- **Đối chiếu số liệu** (kiểm tra ngày 17/11/2026 bằng notebook 02):
  - Bảng bài tập có 837 dòng nhưng chỉ 835 tên khác nhau (2 bài bị lặp), khớp với con số 835 khái niệm của đề bài.
  - 722 bài có tương tác trong nhật ký, khớp với con số 722 câu hỏi trong các bài tổng quan. Như vậy 113 nút chỉ có trong đồ thị.
  - Đồ thị chuyên gia gồm 979 cạnh, 59 nút cô lập, 69 thành phần liên thông yếu.
  - Đồ thị **không phải DAG**: có một chu trình gồm 3 bài về căn thức. Ngoài ra có 2 bài tự ghi chính mình làm tiên quyết; các vòng tự thân này đã bị loại.
  - Sau tiền xử lý: 191.874 người học, 25.853.995 tương tác, tỉ lệ trả lời đúng 82,9%, 281.183 đoạn chuỗi.
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
- **Bảng 1:** AUC, ACC, RMSE của DKT, AKT, GKT-Random, GKT-Stat, GKT-Expert. GKT-Dense bị loại vì vượt bộ nhớ T4; simpleKT để lại cho hướng phát triển.
- **Bảng 2:** nghiên cứu cắt bỏ về cách xử lý nút cô lập và lan truyền một chiều so với hai chiều.
- **Bảng 3:** AUC trên tập khái niệm giữ lại (RQ3).
- **Bảng 4:** AUC theo nhóm khái niệm (cô lập, bậc thấp, bậc cao trong đồ thị chuyên gia; cô lập trong đồ thị thống kê).
- **Kiểm định DeLong ghép cặp** cho hiệu AUC, trên toàn tập kiểm tra và trên khái niệm giữ lại.
- **Hình:** xem mục Kế hoạch hình (Hình 1–5).
- **Case study:** một người học trong tập kiểm tra làm bài trên một chuỗi tiên quyết. So sánh mức thành thạo dự đoán theo từng bước của GKT-Expert, GKT-Stat và DKT (Hình 5).

### 4.4. Đối sánh và thảo luận (trả lời các câu hỏi phản biện)
- **So với AKT:** đồ thị chuyên gia có bù được khoảng cách hiệu năng so với một mô hình chú ý mạnh hay không?
- **Trường hợp GKT-Stat đạt AUC cao hơn GKT-Expert:**
  - Kết quả này không đủ để phủ nhận giá trị của tri thức sư phạm.
  - Đồ thị thống kê có thể chỉ khớp tốt hơn với chính phân phối hành vi mà tập đánh giá đo lường. Đây là một dạng rò rỉ thông tin gián tiếp.
  - Kiểm chứng bằng kết quả trên tập khái niệm giữ lại: nếu khoảng cách thu hẹp hoặc đảo chiều thì giả thuyết rò rỉ được ủng hộ.
- **Khả năng khái quát hóa (RQ4):** chưa thực hiện, nằm ngoài phạm vi kết quả; không đưa ra khẳng định về MOOCCubeX.
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

## Kế hoạch hình

| Hình | Mã hình | Loại | Nội dung và mục tiêu | Nguồn số liệu | Tỉ lệ | Đặt ở mục |
|---|---|---|---|---|---|---|
| 1 | `fig_kien_truc_gkt_hai_chieu` | sơ đồ | **Sơ đồ kiến trúc GKT lan truyền hai chiều:** tương tác `(q_t, a_t)` → biểu diễn nhúng → `f_self` cho nút được trả lời; `f_fwd` tới khái niệm phụ thuộc, `f_bwd` tới khái niệm tiên quyết → cổng xóa–thêm → GRU → dự đoán `P(a_{t+1})`. Thể hiện rõ hai chiều cạnh và việc chỉ cập nhật nút lân cận. | mục 3.3 | 16:9 | 3.3 |
| 2 | `fig_do_thi_chuyen_gia_va_thong_ke` | biểu đồ | So sánh cấu trúc đồ thị: phân phối bậc của đồ thị chuyên gia, thống kê và ngẫu nhiên; số nút cô lập; Jaccard có hướng ≈ 0,01. Kèm biểu đồ hộp giá trị gia tăng của cạnh chuyên gia theo chiều xuôi, chiều ngược và của mọi cặp. Mục tiêu: chứng minh hai nguồn tri thức tách rời. | notebook 03 | 3:2 | 3.4 |
| 3 | `fig_duong_cong_hoc` | biểu đồ | AUC kiểm định theo epoch của 5 mô hình. Mục tiêu: so tốc độ hội tụ và điểm dừng sớm. | notebook 05 (`hinh1_duong_cong_hoc.png`) | 16:9 | 4.3 |
| 4 | `fig_auc_khai_niem_giu_lai` | biểu đồ | Cột ghép AUC trên toàn tập kiểm tra và trên khái niệm giữ lại cho từng mô hình. Mục tiêu: trả lời RQ3 (rò rỉ gián tiếp). | notebook 05 (`hinh2_auc_giu_lai.png`) | 16:9 | 4.4 |
| 5 | `fig_case_study_nguoi_hoc` | biểu đồ | Case study: một người học trong tập kiểm tra làm bài trên một chuỗi tiên quyết. Bản đồ nhiệt mức thành thạo dự đoán theo từng bước cho khái niệm tiên quyết và phụ thuộc, so sánh GKT-Expert, GKT-Stat và DKT; đánh dấu kết quả thật. Mục tiêu: minh họa cách lan truyền theo cạnh tiên quyết. | notebook 05 (`hinh3_case_study.png`) | 16:9 | 4.5 |

---

## Phụ lục

### Phụ lục A. Siêu tham số

| Mô hình | Tham số chính | Cỡ lô | Tối ưu |
|---|---|---|---|
| GKT (cả ba đồ thị) | ẩn 32, nhúng 32, dropout 0,1, chuẩn hóa theo hàng, tự cập nhật nút cô lập | 128 | Adam, tốc độ học 10⁻³, tối đa 20 epoch, dừng sớm sau 3 epoch |
| DKT | nhúng 64, LSTM 64, dropout 0,1 | 128 | như trên |
| AKT | `d_model` 256, 4 khối, 8 head, dropout 0,2 (cài đặt pyKT) | 64 | như trên |
| Đồ thị thống kê | ngưỡng thành thạo 0,5; hỗ trợ tối thiểu 50; ưu tiên 1 cạnh vào mỗi nút; 979 cạnh | — | — |

### Phụ lục B. Đặc trưng và bất thường của dữ liệu
- Bảng bài tập có 837 dòng nhưng chỉ 835 tên khác nhau (`matrix_app_fruit_oil`, `matrix_mul_two` bị lặp); 113 khái niệm không có tương tác.
- Đồ thị chuyên gia gồm 979 cạnh, 59 nút cô lập, 38 nút gốc, 69 thành phần liên thông yếu (thành phần lớn nhất có 744 nút).
- Chu trình: `adding_and_subtracting_radicals` → `radical_multiplication_and_division` → `simplifying_radicals` → quay lại bài đầu. Vòng tự thân bị loại: `proportions_1`, `number_sense_length_l1`.

### Phụ lục C. Tập khái niệm giữ lại
- 108 khái niệm, phân bố theo lĩnh vực (`area`) và chủ đề (`topic`) lấy từ `heldout_concepts.csv`.

### Phụ lục D. Tái lập và cấu trúc kho mã
- Notebook 01–05 chạy trên Colab, cấu trúc kho mã như mục 3.5, 33 ca kiểm thử đơn vị.
- Sản phẩm đầu ra: báo cáo LaTeX (PDF), báo cáo Word (`.docx`), slide seminar (`.pptx`) có case study.

### Phụ lục E. Bảng đối chiếu thuật ngữ

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
19. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
20. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP 2014*.
21. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
22. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
23. Zhang, J., Shi, X., King, I., & Yeung, D.-Y. (2017). Dynamic Key-Value Memory Networks for Knowledge Tracing. *WWW 2017*. (nguồn của cổng xóa–thêm)
24. DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
25. Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
