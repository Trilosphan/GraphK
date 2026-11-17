# Báo cáo Đồ án 8 — Truy vết tri thức dựa trên đồ thị với đồ thị tiên quyết tường minh

> Môn học: Học máy (Machine Learning) · Mã nguồn: https://github.com/Trilosphan/GraphK
> Thực nghiệm: Google Colab, GPU NVIDIA T4 · Dữ liệu: Junyi Academy (bản 01/2015)
> Thuật ngữ theo bảng đối chiếu trong `docs/DAN_Y.md`.

<!-- Bảng và số liệu được điền tự động bởi report/dien_so_lieu.py từ report/data/ket_qua.json. Các đoạn ⟨…⟩ là phần nhận xét viết sau khi có kết quả. -->

---

## Tóm tắt

Đồ án nghiên cứu việc đưa đồ thị tiên quyết do chuyên gia sư phạm xác nhận vào mô hình truy vết tri thức. Mô hình trọng tâm là GKT (Graph-based Knowledge Tracing) được tùy biến với cơ chế lan truyền hai chiều trên đồ thị có hướng: chiều xuôi từ khái niệm tiên quyết tới khái niệm phụ thuộc, và chiều ngược lại. Để tách bạch đóng góp của tri thức chuyên gia và tri thức thống kê, đồ án so sánh cùng một kiến trúc GKT trên ba loại đồ thị: đồ thị chuyên gia, đồ thị thống kê xây dựng từ hành vi làm bài, và đồ thị ngẫu nhiên. Các mô hình cơ sở không dùng đồ thị là DKT và AKT. Để kiểm soát rò rỉ thông tin gián tiếp, đồ thị thống kê chỉ được xây dựng từ tập huấn luyện, và một tập khái niệm giữ lại không có cạnh thống kê nào chạm vào.

⟨KẾT QUẢ: một đến hai câu về phát hiện chính.⟩

---

## 1. Giới thiệu

### 1.1. Bối cảnh

Trí tuệ nhân tạo trong giáo dục (AIED) hướng tới các hệ thống gia sư thông minh có khả năng thích ứng với từng người học. Thành phần cốt lõi của các hệ thống này là mô hình người học: mô hình ước lượng mức độ thành thạo của người học đối với từng khái niệm, làm cơ sở để chọn bài tập kế tiếp và lộ trình học tập.

Junyi Academy, một nền tảng học trực tuyến của Đài Loan phát triển từ mã nguồn của Khan Academy, có sẵn bản đồ tri thức trong đó quan hệ "phải học trước" giữa các bài tập đã được đội ngũ sư phạm xác nhận. Các mô hình truy vết tri thức tuần tự như DKT và DKVMN không sử dụng cấu trúc này. Đồ án đặt câu hỏi: đưa trực tiếp đồ thị tiên quyết vào cơ chế lan truyền của mô hình có cải thiện dự đoán hay không, và cải thiện đó đến từ tri thức chuyên gia hay chỉ từ việc có thêm cấu trúc.

### 1.2. Bài toán truy vết tri thức

Cho chuỗi tương tác `(q_1, a_1), …, (q_t, a_t)` của một người học, trong đó `q_i` là khái niệm (bài tập) và `a_i ∈ {0, 1}` là kết quả trả lời ở lần thử đầu tiên. Bài toán truy vết tri thức ước lượng trạng thái tri thức ẩn của người học và dự đoán xác suất trả lời đúng ở bước kế tiếp, `P(a_{t+1} = 1 | q_{t+1}, q_{1..t}, a_{1..t})`. Hiệu năng được đo bằng AUC-ROC của dự đoán này trên những người học chưa xuất hiện khi huấn luyện.

Trong bài toán của đồ án, mô hình còn nhận đầu vào là đồ thị có hướng `G = (V, E)` với 835 nút. Cạnh `u → v` cho biết khái niệm `u` là tiên quyết của khái niệm `v`.

### 1.3. Câu hỏi nghiên cứu

- **RQ1:** GKT dùng đồ thị chuyên gia có dự đoán tốt hơn các mô hình không dùng đồ thị (DKT, AKT) không?
- **RQ2:** Đóng góp riêng của tri thức chuyên gia so với tri thức thống kê là bao nhiêu, khi giữ nguyên kiến trúc và số cạnh?
- **RQ3:** Nếu đồ thị thống kê cho AUC cao hơn, đó là giá trị thật hay rò rỉ thông tin gián tiếp? Thiết kế đánh giá nào cho phép so sánh công bằng?
- **RQ4:** Khả năng khái quát hóa sang MOOCCubeX. Đồ án **chưa thực hiện** câu hỏi này do giới hạn tài nguyên tính toán (xem mục 6).

### 1.4. Đóng góp

1. Cài đặt GKT lan truyền hai chiều trên đồ thị có hướng, biểu diễn lân cận dạng danh sách cạnh nén (CSR) để chi phí mỗi bước tỉ lệ với bậc thực tế của nút. Hai cách xử lý nút cô lập được so sánh bằng nghiên cứu cắt bỏ.
2. Một giao thức so sánh tách bạch gồm đồ thị chuyên gia, đồ thị thống kê (cùng số cạnh) và đồ thị ngẫu nhiên (cùng số cạnh) trên cùng kiến trúc.
3. Một giao thức chống rò rỉ: đồ thị thống kê chỉ dựng từ tập huấn luyện của từng phần, và tập khái niệm giữ lại không có cạnh thống kê nào chạm vào.
4. Phân tích dữ liệu cho thấy đồ thị chuyên gia của Junyi có một chu trình và hai vòng tự thân. Tương quan thống kê giữa các cặp khái niệm không phân biệt được chiều tiên quyết.
5. Quy trình xử lý tái lập được gồm 5 notebook Colab, 31 ca kiểm thử đơn vị (có kiểm thử chống rò rỉ theo thời gian cho cả ba mô hình), và cơ chế học tiếp từ điểm kiểm tra khi phiên Colab bị ngắt.

---

## 2. Các nghiên cứu liên quan

Danh sách đầy đủ và phân tích từng công trình (phương pháp, dữ liệu, ưu điểm, hạn chế, kết quả) nằm ở mục 2 của `docs/DAN_Y.md`. Mục này tóm tắt các nhóm công trình và khoảng trống mà đồ án nhắm tới.

**Mô hình không dùng đồ thị.**
- AKT (Ghosh và cộng sự, KDD 2020) dùng Transformer với chú ý suy giảm đơn điệu theo khoảng cách thời gian, và là một trong các mô hình có AUC cao nhất trên nhiều tập dữ liệu chuẩn.
- simpleKT (Liu và cộng sự, ICLR 2023) cho thấy nhiều mô hình phức tạp không vượt được một mô hình cơ sở đơn giản khi được đánh giá theo giao thức chuẩn hóa của pyKT (NeurIPS 2022).

**Mô hình dùng đồ thị suy ra từ dữ liệu hoặc nhãn khái niệm.**
- JKT (Information Sciences, 2021) và Bi-CLKT (Knowledge-Based Systems, 2022) xây dựng đồ thị bài tập–bài tập và khái niệm–khái niệm từ thống kê đồng xuất hiện.
- DGEKT (ACM TOIS, 2024) kết hợp siêu đồ thị bài tập–khái niệm với đồ thị chuyển tiếp có hướng.
- GIKT (ECML-PKDD 2020) dùng GCN trên đồ thị câu hỏi–kỹ năng.
- HHSKT (Expert Systems with Applications, 2023) dùng mạng nơ-ron đồ thị không thuần nhất trên đồ thị người học–câu hỏi–khái niệm.
- Không công trình nào ở trên dùng đồ thị tiên quyết của chuyên gia.

**Mô hình có quan hệ tiên quyết.**
- SKT (ICDM 2020) lan truyền ảnh hưởng từng phần theo quan hệ có hướng và đồng bộ theo quan hệ vô hướng; đây là căn cứ cho cơ chế lan truyền hai chiều của đồ án.
- PSI-KT (ICLR 2024) suy ra đồng thời đặc trưng người học và đồ thị tiên quyết dùng chung từ dữ liệu làm bài.
- Cả hai đều ước lượng cấu trúc từ dữ liệu, nên không tách riêng được đóng góp của tri thức chuyên gia.

**Khoảng trống.** Chưa có đánh giá nào so sánh đồ thị chuyên gia với đồ thị thống kê trong cùng một kiến trúc GKT, với cùng số cạnh, kèm giao thức kiểm soát rò rỉ trên tập khái niệm chưa từng được dùng để xây dựng đồ thị thống kê.

---

## 3. Phương pháp

### 3.1. Tiền xử lý dữ liệu (notebook 02)

- **Nguồn dữ liệu:** bộ dữ liệu Junyi Academy (PSLC DataShop, `datasetId=1198`), tải từ bản sao của USTC, có kiểm tra SHA256.
- **Bảng bài tập:**
  - Bảng có 837 dòng nhưng chỉ 835 tên khác nhau: hai bài `matrix_app_fruit_oil` và `matrix_mul_two` bị lặp, hai bản lặp có cùng danh sách tiên quyết. Đồ án giữ bản có ngày tạo mới nhất. Đây là nguồn gốc của con số 835 khái niệm trong đề bài.
  - Mỗi tên bài tập được ánh xạ sang `node_id` theo thứ tự từ điển, để kết quả tất định.
- **Đồ thị chuyên gia:**
  - Cột `prerequisites` của bài X liệt kê các bài tiên quyết P, mỗi P sinh một cạnh P → X.
  - Hai vòng tự thân (`proportions_1`, `number_sense_length_l1`) và 3 cạnh trùng bị loại.
  - Kết quả có 979 cạnh, 59 nút cô lập, 38 nút gốc, 69 thành phần liên thông yếu.
  - Đồ thị **không phải DAG**: có một chu trình gồm ba bài về căn thức (`adding_and_subtracting_radicals` → `radical_multiplication_and_division` → `simplifying_radicals` → quay lại bài đầu). Lan truyền trên GNN không đòi hỏi đồ thị không chu trình, nên đồ án giữ nguyên chu trình này và ghi nhận đây là một điểm không nhất quán của dữ liệu chuyên gia.
- **Nhật ký tương tác:**
  - Gồm 25.925.992 dòng. Chỉ 722/835 bài có tương tác; 113 nút chỉ có trong đồ thị.
  - Sắp xếp theo `(user_id, time_done)` và loại người học có dưới 3 tương tác. Còn lại 191.874 người học và 25.853.995 tương tác, tỉ lệ trả lời đúng 82,9%.
  - Chuỗi dài được chia thành các đoạn liên tiếp tối đa 200 bước theo quy ước của pyKT, được 281.183 đoạn có độ dài trung bình 91,9.
- **Phân chia theo người học:** 20% người học (38.375) làm tập kiểm tra chung. Số còn lại chia thành 5 phần, mỗi phần khoảng 30.700 người học. Ở phần k, phần k là tập kiểm định và các phần khác là tập huấn luyện.

### 3.2. Đồ thị đối chứng và tập khái niệm giữ lại (notebook 03)

**Tập khái niệm giữ lại.** Chọn ngẫu nhiên 15% số khái niệm có tương tác, được 108 khái niệm (257/979 cạnh chuyên gia chạm vào chúng). Khái niệm giữ lại vẫn xuất hiện trong dữ liệu huấn luyện và vẫn có cạnh trong đồ thị chuyên gia. Điểm khác duy nhất là **không có cạnh thống kê nào chạm vào chúng**, nên trên các khái niệm này đồ thị thống kê không thể khớp với phân phối hành vi.

**Đồ thị thống kê (GKT-Stat).**
- Người học `u` thành thạo khái niệm `c` khi tỉ lệ trả lời đúng của `u` trên `c` ≥ 0,5.
- Với mỗi cặp có hướng A → B, chỉ xét người học đã làm cả hai khái niệm, và tính giá trị gia tăng (added value):

  `AV(A → B) = P(thành thạo B | thành thạo A) − P(thành thạo B)`.

- Chỉ giữ cặp có ít nhất 50 người học thành thạo A và đã làm B.
- Chọn đúng 979 cạnh (bằng đồ thị chuyên gia). Trước hết mỗi khái niệm nhận cạnh vào có AV lớn nhất, phần còn lại lấy theo AV toàn cục. Cách chọn thuần túy theo AV toàn cục làm cạnh dồn vào khoảng 190 nút và để lại 642 nút cô lập, nên đã bị thay thế.
- Đồ thị của phần k chỉ dùng người học thuộc tập huấn luyện của phần k.

**Đồ thị ngẫu nhiên (GKT-Random):** 979 cạnh có hướng chọn ngẫu nhiên đều, không có vòng tự thân, đóng vai trò đối chứng âm cho câu hỏi "chỉ cần có cấu trúc là đủ?".

![Hình 2. Bên trái: phân phối bậc (vào + ra) của đồ thị chuyên gia, thống kê và ngẫu nhiên. Bên phải: giá trị gia tăng của cạnh chuyên gia theo chiều xuôi, chiều ngược, của cạnh thống kê và của mọi cặp đủ hỗ trợ (tập huấn luyện phần 0).](report/figures/hinh2_do_thi.png)

**Đặc trưng so sánh của các đồ thị (phần 0):**

| Đồ thị | Cạnh | Nút cô lập | Thành phần liên thông yếu | Jaccard có hướng với đồ thị chuyên gia |
|---|---|---|---|---|
| Chuyên gia | 979 | 59 | 69 | 1,000 |
| Thống kê | 979 | 331 | 332 | 0,010 |
| Ngẫu nhiên | 979 | 67 | 76 | 0,000 |

- Đồ thị thống kê và đồ thị chuyên gia gần như không trùng cạnh, nên chúng là hai nguồn tri thức tách rời.
- Trên tập huấn luyện của phần 0, chỉ 637/979 cạnh chuyên gia có đủ hỗ trợ để tính AV.
- Trung vị AV của cạnh chuyên gia theo chiều xuôi là 0,020, theo chiều ngược là 0,022, còn của mọi cặp là 0,013. Cạnh chuyên gia có tương quan nhỉnh hơn mặt bằng chung, nhưng dữ liệu hành vi **không phân biệt được chiều tiên quyết**.
- 221 trong số 331 nút cô lập của đồ thị thống kê là buộc phải cô lập: 108 khái niệm giữ lại và 113 khái niệm không có tương tác.

### 3.3. Mô hình GKT lan truyền hai chiều

![Hình 1. Kiến trúc GKT lan truyền hai chiều. (a) Đồ thị tiên quyết quanh khái niệm i đang được trả lời. (b) Một bước cập nhật: thông điệp tự thân cho i, thông điệp ngược tới khái niệm tiên quyết p, thông điệp xuôi tới khái niệm phụ thuộc d; cổng xóa–thêm rồi GRU, chỉ cập nhật i và lân cận.](report/figures/hinh1_kien_truc.png)

Mỗi khái niệm `j` có một trạng thái ẩn `h_j ∈ R^32`. Tại bước `t`, khi người học trả lời khái niệm `i` với kết quả `a_t`:

1. **Biểu diễn tương tác:** `x_t = E_resp[2·i + a_t]` và `s = [h_i ; x_t]`.
2. **Thông điệp tự thân:** `m_i = f_self(s)`.
3. **Thông điệp lân cận theo chiều cạnh:**
   - mỗi khái niệm phụ thuộc `j` (cạnh `i → j`) nhận `m_j = w_ij · f_fwd([s ; h_j ; E_node[j]])`;
   - mỗi khái niệm tiên quyết `j` (cạnh `j → i`) nhận `m_j = w_ij · f_bwd([s ; h_j ; E_node[j]])`;
   - trọng số `w_ij` bằng nghịch đảo số lân cận cùng chiều của `i` (chuẩn hóa theo hàng, bậc 0 không sinh phép chia).
   - Lập luận cho hai hàm riêng: trả lời đúng bài Tích phân là bằng chứng mạnh về việc đã thành thạo Đạo hàm, còn chiều ngược lại yếu hơn. Hai chiều cần tham số riêng.
4. **Cập nhật:** thông điệp qua cổng xóa–thêm `m' = m − w_c·σ(W_e m)⊙m + w_c·tanh(W_a m)`, rồi `h_j ← GRU(m'_j, h_j)`. Chỉ khái niệm `i` và các lân cận của nó được cập nhật.
5. **Dự đoán:** `P(a_{t+1} = 1) = σ(W_o h_{q_{t+1}} + b)`.

Các hàm `f_self`, `f_fwd`, `f_bwd` là MLP hai tầng với ReLU và dropout 0,1. Mô hình có 129.701 tham số.

**Cài đặt hiệu quả.**
- Lân cận được lưu dạng danh sách cạnh nén (CSR), không có ô đệm. Tại mỗi bước, các cạnh của những khái niệm đang được trả lời trong lô được trải phẳng, nên chi phí tỉ lệ với tổng bậc thực tế.
- Với bảng đệm, đồ thị thống kê có nút trung tâm bậc ra 84 buộc mọi bước tính trên 103 cột. Danh sách nén giảm thời gian mỗi lô của GKT-Stat trên CPU từ 9,97 s xuống 1,35 s, ngang với các đồ thị khác.

**Xử lý nút cô lập** (dàn ý mục 3.2), so sánh bằng nghiên cứu cắt bỏ:
- (a) **tự cập nhật** (mặc định): nút cô lập chỉ nhận `f_self`, tương đương thêm vòng tự thân;
- (b) **nút gốc ảo**: thêm một nút nối tới mọi khái niệm có bậc vào 0, biểu diễn năng lực chung.
- Chuẩn hóa an toàn được bảo đảm trong cả hai cách: nút bậc 0 không có lân cận nên không sinh phép chia.

**Mô hình cơ sở.**
- DKT: LSTM 64 chiều trên biểu diễn nhúng của cặp (khái niệm, kết quả).
- AKT: chép nguyên văn cài đặt của pyKT (giấy phép MIT, commit `8e23db4`), với `d_model = 256`, 4 khối, dropout 0,2.
- Mọi mô hình trả về logit cho `a_{2..T}`. Kiểm thử đơn vị xác nhận dự đoán ở bước `t` không đổi khi sửa các kết quả từ bước `t+1` trở đi, tức không rò rỉ theo thời gian.

### 3.4. Huấn luyện (notebook 04)

- **Tối ưu:** Adam với tốc độ học 10⁻³, hàm mất mát entropy chéo nhị phân trên các vị trí hợp lệ, cắt chuẩn gradient tại 1,0 cho mọi mô hình. Lần chạy đầu không cắt gradient, GKT-Stat phân kỳ (mất mát thành NaN) giữa epoch 2, nhiều khả năng do các nút trung tâm bậc cao của đồ thị thống kê. Vì vậy mọi mô hình, kể cả GKT-Expert, được huấn luyện lại với cùng giao thức cắt gradient. Cắt gradient vẫn không đủ: GKT-Stat tiếp tục cho dự đoán NaN ở epoch 2. Nguyên nhân là đồ thị thống kê có 130 cặp cạnh hai chiều (A → B và B → A). Khi trả lời A, nút B vừa là khái niệm phụ thuộc vừa là khái niệm tiên quyết, nên bị cập nhật hai lần trong cùng một phép ghi chỉ số (`index_put` với chỉ số trùng cho kết quả không xác định). Bản cài đặt cuối gộp hai chiều vào một lân cận, với thông điệp `w_fwd·f_fwd + w_bwd·f_bwd`, và bỏ bước cập nhật khi chuẩn gradient không hữu hạn. Trên đồ thị chuyên gia và đồ thị ngẫu nhiên (không có cặp hai chiều), đầu ra của bản cuối trùng khớp tuyệt đối với bản trước, nên chỉ GKT-Stat phải huấn luyện lại.
- **Cỡ lô:** 128; riêng AKT dùng 64 vì vượt 15 GB bộ nhớ của T4 ở cỡ lô 128.
- **Gom lô:** các đoạn có độ dài gần nhau được gom vào cùng lô để giảm số bước đệm, và thứ tự lô được xáo trộn.
- **Dừng sớm:** tối đa 10 epoch, dừng sau 3 epoch AUC kiểm định không cải thiện. Giới hạn 10 epoch được đặt sau khi quan sát GKT-Expert: từ epoch 4, AUC kiểm định chỉ tăng khoảng 0,0005 mỗi epoch, nên chạy tới 20 epoch sẽ tốn gấp đôi thời gian GPU mà gần như không đổi kết quả. Mô hình tốt nhất theo AUC kiểm định được đánh giá trên tập kiểm tra chung.
- **Điểm kiểm tra:** lưu lên Google Drive sau mỗi epoch; chạy lại cùng cấu hình thì học tiếp từ epoch gần nhất.
- Thời gian một epoch trên T4: GKT khoảng 12 phút, AKT khoảng 10,7 phút, DKT khoảng 0,2 phút.

---

## 4. Thực nghiệm

### 4.1. Thiết lập

- **Dữ liệu:** như mục 3.1. Kết quả trong báo cáo là của **phần 0**: tập huấn luyện 122.799 người học (179.506 đoạn), tập kiểm định 30.700 người học, tập kiểm tra 38.375 người học. Bốn phần còn lại chưa chạy vì giới hạn đơn vị tính toán Colab (ước tính thêm 30–45 giờ GPU).
- **Độ đo:**
  - Độ đo chính là AUC-ROC; độ đo phụ là ACC (ngưỡng 0,5) và RMSE.
  - AUC được tính riêng trên tập khái niệm giữ lại và trên các khái niệm còn lại.
  - AUC theo nhóm khái niệm: theo bậc trong đồ thị chuyên gia, và theo việc khái niệm có bị cô lập trong đồ thị thống kê hay không.
- **Kiểm định:**
  - Dùng kiểm định DeLong ghép cặp (Sun & Xu, 2014) cho hiệu AUC giữa hai mô hình trên cùng các dự đoán.
  - Mọi mô hình dự đoán trên tập kiểm tra theo cùng một thứ tự tất định, nên các dự đoán ghép cặp được với nhau.
  - Kiểm định giả định các dự đoán độc lập, trong khi thực tế các tương tác của cùng một người học có tương quan. Với hàng triệu dự đoán, giá trị p rất nhỏ ngay cả khi hiệu AUC nhỏ, nên báo cáo nhấn mạnh độ lớn của hiệu.

### 4.2. Kết quả chính (Bảng 1)

**Bảng 1.** Kết quả trên tập kiểm tra (38.375 người học), phần 0. AUC cao nhất in đậm.

<!-- BANG1 -->

⟨NHẬN XÉT BẢNG 1⟩

![Hình 3. AUC kiểm định theo epoch của năm mô hình (phần 0).](report/figures/hinh3_duong_cong_hoc.png)

### 4.3. Nghiên cứu cắt bỏ (Bảng 2)

**Bảng 2.** Các biến thể của GKT-Expert: xử lý nút cô lập (tự cập nhật hay nút gốc ảo) và lan truyền hai chiều hay một chiều.

<!-- BANG2 -->

⟨NHẬN XÉT BẢNG 2⟩

### 4.4. Khái niệm giữ lại và nhóm khái niệm (Bảng 3, Bảng 4)

**Bảng 3.** AUC trên 108 khái niệm giữ lại (không có cạnh thống kê nào chạm vào) và trên các khái niệm còn lại.

<!-- BANG3 -->

![Hình 4. AUC trên toàn tập kiểm tra và trên khái niệm giữ lại theo từng mô hình.](report/figures/hinh4_auc_giu_lai.png)

**Bảng 4.** AUC theo nhóm khái niệm đích: theo bậc trong đồ thị chuyên gia, và theo việc khái niệm có bị cô lập trong đồ thị thống kê hay không.

<!-- BANG4 -->

⟨NHẬN XÉT BẢNG 3, BẢNG 4⟩

### 4.5. Kiểm định ghép cặp

**Bảng 5.** Kiểm định DeLong ghép cặp cho hiệu AUC (mô hình 1 − mô hình 2) trên cùng các dự đoán của tập kiểm tra.

<!-- DELONG -->

⟨NHẬN XÉT KIỂM ĐỊNH⟩

### 4.6. Case study

<!-- CASE -->

![Hình 5. Case study: mức thành thạo dự đoán sau từng bước (màu) cho các khái niệm trong chuỗi tiên quyết, theo GKT-Expert, GKT-Stat và DKT. Dấu o: trả lời đúng, x: trả lời sai trên khái niệm của hàng đó.](report/figures/hinh5_case_study.png)

⟨NHẬN XÉT CASE STUDY⟩

---

## 5. Thảo luận

⟨KẾT QUẢ: trả lời RQ1–RQ3 và hai câu hỏi phản biện của đề bài.⟩

---

## 6. Hạn chế và hướng phát triển

- **Một phần dữ liệu, một hạt giống ngẫu nhiên:** chưa có độ lệch chuẩn qua các phần. Cần chạy 4 phần còn lại để báo cáo trung bình ± độ lệch chuẩn.
- **RQ4 (MOOCCubeX) chưa thực hiện:** cần ánh xạ khái niệm và xây dựng nhật ký làm bài theo khái niệm của MOOCCubeX.
- **Đồ thị Junyi là đồ thị bài tập,** chưa hẳn là đồ thị khái niệm, và chỉ có một loại cạnh.
- **Kiểm định DeLong bỏ qua tương quan trong cùng người học.** Kiểm định bootstrap theo người học sẽ cho khoảng tin cậy thận trọng hơn.
- **Hướng phát triển:**
  - học cấu trúc đồ thị với đồ thị chuyên gia làm phân phối tiên nghiệm;
  - đồ thị nhiều loại cạnh, theo SKT;
  - kết hợp cơ chế chú ý của AKT với lan truyền trên đồ thị.

---

## 7. Tái lập kết quả

Chạy lần lượt trên Google Colab, mỗi notebook tự lấy mã mới nhất từ GitHub và lưu kết quả lên Google Drive:

| Notebook | Nội dung | Thời gian (T4) |
|---|---|---|
| 01 | tải và giải nén dữ liệu Junyi | khoảng 20 phút |
| 02 | tiền xử lý | khoảng 3 phút |
| 03 | đồ thị đối chứng, tập khái niệm giữ lại | khoảng 1 phút |
| 04 | huấn luyện 5 mô hình chính và 2 thí nghiệm cắt bỏ (`QUICK = False`, `RUN_ABLATIONS = True`) | khoảng 12 giờ cho phần 0 |
| 05 | phân tích, bảng, hình, kiểm định | khoảng 5 phút |

Kiểm thử trên máy: `python -m pytest` (31 ca).

## Phụ lục

### Phụ lục A. Siêu tham số

| Mô hình | Tham số chính | Cỡ lô | Tối ưu |
|---|---|---|---|
| GKT (cả ba đồ thị) | ẩn 32, nhúng 32, dropout 0,1, chuẩn hóa theo hàng, tự cập nhật nút cô lập | 128 | Adam, tốc độ học 10⁻³, cắt chuẩn gradient 1,0, tối đa 10 epoch, dừng sớm sau 3 epoch |
| DKT | nhúng 64, LSTM 64, dropout 0,1 | 128 | như trên |
| AKT | `d_model` 256, 4 khối, 8 head, dropout 0,2 (cài đặt pyKT) | 64 | như trên |
| Đồ thị thống kê | ngưỡng thành thạo 0,5; hỗ trợ tối thiểu 50; ưu tiên 1 cạnh vào mỗi nút; 979 cạnh | — | — |

### Phụ lục B. Cấu trúc kho mã

| Thành phần | Nội dung |
|---|---|
| `src/graphk/preprocess.py` | tiền xử lý, đồ thị chuyên gia, thống kê đồ thị (DAG, chu trình) |
| `src/graphk/graphs.py` | đồ thị thống kê (giá trị gia tăng), đồ thị ngẫu nhiên, khái niệm giữ lại |
| `src/graphk/models/` | GKT hai chiều (CSR), DKT, AKT (chép từ pyKT) |
| `src/graphk/train.py` | huấn luyện, dừng sớm, điểm kiểm tra, đánh giá |
| `src/graphk/analysis.py` | bảng tổng hợp, AUC theo nhóm, DeLong, case study |
| `notebooks/01–06` | quy trình trên Colab, sinh từ `scripts/tao_notebook.py` |
| `report/` | sinh báo cáo LaTeX, Word và slide seminar |

## Tài liệu tham khảo

Xem danh mục đầy đủ ở cuối `docs/DAN_Y.md`, bổ sung:

- Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
