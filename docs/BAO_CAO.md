# Truy vết tri thức dựa trên đồ thị với đồ thị tiên quyết tường minh: đóng góp của tri thức chuyên gia so với tri thức thống kê

<!-- Nguồn nội dung của báo cáo (LaTeX IEEEtran, Word, slide). Bảng và số liệu do report/dien_so_lieu.py điền từ report/data/ket_qua.json;
trích dẫn dạng [@…] được đánh số kiểu IEEE theo thứ tự xuất hiện từ report/tai_lieu.json. Các đoạn ⟨…⟩ chờ kết quả còn lại. -->

## Tóm tắt

Các mô hình truy vết tri thức tuần tự bỏ qua bản đồ tiên quyết mà chuyên gia sư phạm đã xây dựng, còn các mô hình dựa trên đồ thị hiện có lại tự suy đồ thị từ dữ liệu, nên chưa tách được giá trị riêng của tri thức chuyên gia. Bài báo đề xuất GKT lan truyền hai chiều trên đồ thị tiên quyết có hướng, với hàm thông điệp riêng cho chiều xuôi (tiên quyết → phụ thuộc) và chiều ngược. Cùng một kiến trúc được huấn luyện trên ba đồ thị có cùng 979 cạnh: đồ thị chuyên gia của Junyi Academy, đồ thị thống kê dựng từ giá trị gia tăng của hành vi làm bài trên tập huấn luyện, và đồ thị ngẫu nhiên. Rò rỉ gián tiếp được kiểm soát bằng 108 khái niệm giữ lại không có cạnh thống kê nào. Trên 191.874 người học và 25,85 triệu tương tác, GKT-Expert đạt AUC {{auc:GKT-Expert}}, cao hơn GKT-Stat ({{auc:GKT-Stat}}) và GKT-Random ({{auc:GKT-Random}}) khoảng 0,003, và lợi thế này giữ nguyên trên khái niệm giữ lại; đồ thị thống kê không tốt hơn đồ thị ngẫu nhiên. Tuy vậy, DKT không dùng đồ thị vẫn đạt AUC cao nhất ({{auc:DKT}}). Phân tích dữ liệu cho thấy đồ thị chuyên gia chứa một chu trình và hai vòng tự thân, gần như không trùng với tương quan hành vi (Jaccard ≈ 0,01), và tương quan hành vi không phân biệt được chiều tiên quyết.

**Từ khóa—**truy vết tri thức, mạng nơ-ron đồ thị, đồ thị tiên quyết, Junyi Academy, rò rỉ dữ liệu, giáo dục thông minh.

## 1. Giới thiệu

Trí tuệ nhân tạo trong giáo dục hướng tới các hệ thống gia sư thông minh thích ứng với từng người học. Thành phần cốt lõi là mô hình người học, ước lượng mức độ thành thạo của người học đối với từng khái niệm để chọn bài tập kế tiếp và lộ trình học tập [@survey1; @survey2]. Truy vết tri thức (knowledge tracing, KT) giải bài toán này từ chuỗi tương tác: từ mô hình xác suất BKT [@bkt], qua mô hình học sâu tuần tự DKT [@dkt] và DKVMN [@dkvmn], đến các mô hình dựa trên cơ chế chú ý như AKT [@akt].

Cho chuỗi tương tác `(q_1, a_1), …, (q_t, a_t)` của một người học, trong đó `q_i` là khái niệm (bài tập) và `a_i ∈ {0, 1}` là kết quả trả lời ở lần thử đầu tiên, KT dự đoán `P(a_{t+1} = 1 | q_{t+1}, q_{1..t}, a_{1..t})`. Hiệu năng được đo bằng AUC-ROC trên những người học chưa xuất hiện khi huấn luyện.

Junyi Academy [@junyi] có sẵn bản đồ tri thức, trong đó quan hệ "phải học trước" giữa 835 bài tập đã được đội ngũ sư phạm xác nhận. Các mô hình tuần tự không dùng cấu trúc này, còn GKT [@gkt] và các biến thể dùng đồ thị lại suy đồ thị từ dữ liệu. Bài báo trả lời ba câu hỏi:

- **RQ1:** GKT dùng đồ thị chuyên gia có dự đoán tốt hơn các mô hình không dùng đồ thị (DKT, AKT) không?
- **RQ2:** Đóng góp riêng của tri thức chuyên gia so với tri thức thống kê là bao nhiêu, khi giữ nguyên kiến trúc và số cạnh?
- **RQ3:** Nếu đồ thị thống kê cho AUC cao hơn, đó là giá trị thật hay rò rỉ thông tin gián tiếp, và đánh giá thế nào cho công bằng?

Khả năng khái quát hóa sang MOOCCubeX [@mooccubex] nằm ngoài phạm vi kết quả của bài báo.

**Đóng góp.** (i) GKT lan truyền hai chiều trên đồ thị có hướng, cài đặt bằng danh sách cạnh nén để chi phí mỗi bước tỉ lệ với bậc thực tế; (ii) giao thức so sánh tách bạch giữa đồ thị chuyên gia, thống kê và ngẫu nhiên cùng số cạnh; (iii) giao thức chống rò rỉ gồm đồ thị thống kê chỉ dựng từ tập huấn luyện và tập khái niệm giữ lại; (iv) phân tích chất lượng đồ thị chuyên gia của Junyi; (v) quy trình tái lập được trên Google Colab với kiểm thử chống rò rỉ theo thời gian.

## 2. Các nghiên cứu liên quan

**KT không dùng đồ thị.** AKT [@akt] dùng Transformer [@transformer] với chú ý suy giảm đơn điệu theo khoảng cách thời gian và thuộc nhóm có AUC cao nhất trên nhiều tập dữ liệu chuẩn. simpleKT [@simplekt] cho thấy nhiều mô hình phức tạp không vượt được một mô hình cơ sở đơn giản khi đánh giá theo giao thức chuẩn hóa của pyKT [@pykt]. Hạn chế chung của nhóm này là quan hệ giữa các khái niệm chỉ được học ngầm.

**KT dùng đồ thị suy từ dữ liệu.** JKT [@jkt] và Bi-CLKT [@biclkt] dựng đồ thị bài tập–bài tập và khái niệm–khái niệm từ thống kê đồng xuất hiện, dùng mạng tích chập đồ thị [@gcn] hoặc học tương phản. DGEKT [@dgekt] kết hợp siêu đồ thị với đồ thị chuyển tiếp có hướng; GIKT [@gikt] dùng GCN trên đồ thị câu hỏi–kỹ năng; HHSKT [@hhskt] dùng GNN không thuần nhất. Vì đồ thị được suy từ chính dữ liệu hành vi, lợi thế của chúng có thể chỉ phản ánh việc khớp phân phối của tập đánh giá; không công trình nào đánh giá trên khái niệm chưa dùng để dựng đồ thị.

**KT có quan hệ tiên quyết.** SKT [@skt] lan truyền ảnh hưởng từng phần theo quan hệ có hướng và đồng bộ theo quan hệ vô hướng, là căn cứ cho cơ chế hai chiều của bài báo. PSI-KT [@psikt] suy ra đồng thời đặc trưng người học và đồ thị tiên quyết dùng chung. GKT gốc [@gkt] chỉ thử đồ thị đầy đủ, đồ thị chuyển tiếp và đồ thị học được. Chưa có đánh giá nào so sánh đồ thị chuyên gia với đồ thị thống kê trong cùng một kiến trúc, cùng số cạnh, kèm kiểm soát rò rỉ.

## 3. Phương pháp

### 3.1. Dữ liệu và đồ thị chuyên gia

Bộ dữ liệu Junyi Academy [@junyi] gồm bảng bài tập 837 dòng nhưng chỉ 835 tên khác nhau (`matrix_app_fruit_oil` và `matrix_mul_two` bị lặp; giữ bản có ngày tạo mới nhất). Cột `prerequisites` của bài X sinh cạnh P → X cho mỗi tiên quyết P. Sau khi loại 2 vòng tự thân (`proportions_1`, `number_sense_length_l1`) và 3 cạnh trùng, đồ thị có 979 cạnh, 59 nút cô lập, 38 nút gốc và 69 thành phần liên thông yếu. Đồ thị **không phải DAG**: có một chu trình gồm ba bài về căn thức (`adding_and_subtracting_radicals` → `radical_multiplication_and_division` → `simplifying_radicals`); lan truyền trên GNN không đòi hỏi đồ thị không chu trình nên chu trình được giữ nguyên.

Nhật ký gồm 25.925.992 tương tác; chỉ 722/835 bài có tương tác. Sau khi loại người học có dưới 3 tương tác còn 191.874 người học và 25.853.995 tương tác (tỉ lệ đúng 82,9%), chia thành 281.183 đoạn tối đa 200 bước theo quy ước của pyKT [@pykt]. Người học được chia: 20% (38.375) làm tập kiểm tra chung, phần còn lại chia 5 phần; ở phần k, phần k là tập kiểm định.

### 3.2. GKT lan truyền hai chiều

![Hình 1. Kiến trúc GKT lan truyền hai chiều. (a) Đồ thị tiên quyết quanh khái niệm i đang được trả lời. (b) Một bước cập nhật: thông điệp tự thân cho i, thông điệp ngược tới khái niệm tiên quyết p, thông điệp xuôi tới khái niệm phụ thuộc d; cổng xóa–thêm rồi GRU, chỉ cập nhật i và lân cận.](report/figures/hinh1_kien_truc.png)

Mỗi khái niệm `j` có trạng thái ẩn `h_j ∈ R^32` (Hình 1). Tại bước `t`, khi người học trả lời khái niệm `i` với kết quả `a_t`:

1. **Biểu diễn tương tác:** `x_t = E_resp[2·i + a_t]`, `s = [h_i ; x_t]`.
2. **Thông điệp tự thân:** `m_i = f_self(s)`.
3. **Thông điệp lân cận:** với mỗi lân cận `j`, `m_j = w_fwd·f_fwd([s ; h_j ; e_j]) + w_bwd·f_bwd([s ; h_j ; e_j])`, trong đó `w_fwd > 0` khi có cạnh `i → j` (j phụ thuộc i) và `w_bwd > 0` khi có cạnh `j → i` (j là tiên quyết của i); trọng số bằng nghịch đảo số lân cận cùng chiều. Hai chiều có tham số riêng vì làm đúng Tích phân là bằng chứng mạnh về Đạo hàm, còn chiều ngược lại yếu hơn [@skt].
4. **Cập nhật:** cổng xóa–thêm `m' = m − w_c·σ(W_e m)⊙m + w_c·tanh(W_a m)` [@dkvmn], rồi `h_j ← GRU(m'_j, h_j)` [@gru]; chỉ `i` và lân cận được cập nhật.
5. **Dự đoán:** `P(a_{t+1} = 1) = σ(W_o h_{q_{t+1}} + b)`.

`f_self`, `f_fwd`, `f_bwd` là MLP hai tầng (ReLU, dropout 0,1); mô hình có 129.701 tham số. Lân cận được lưu dạng danh sách cạnh nén, mỗi cặp (i, j) đúng một mục: đồ thị thống kê có 130 cặp cạnh hai chiều, nếu không gộp thì một nút bị ghi hai lần trong cùng một phép ghi chỉ số và quá trình huấn luyện cho kết quả NaN. Nút cô lập chỉ nhận `f_self` (tương đương vòng tự thân); phương án thay thế là một nút gốc ảo nối tới mọi khái niệm có bậc vào 0, được so sánh trong nghiên cứu cắt bỏ.

**Mô hình cơ sở.** DKT [@dkt] dùng LSTM [@lstm] 64 chiều; AKT [@akt] dùng cài đặt của pyKT [@pykt] với `d_model = 256`, 4 khối. Kiểm thử đơn vị xác nhận dự đoán ở bước `t` không đổi khi sửa các kết quả từ bước `t+1` trở đi.

### 3.3. Đồ thị đối chứng và chống rò rỉ

**Đồ thị thống kê.** Người học `u` thành thạo khái niệm `c` khi tỉ lệ đúng trên `c` ≥ 0,5. Với cặp có hướng A → B, chỉ xét người học đã làm cả hai, điểm giá trị gia tăng là `AV(A → B) = P(thành thạo B | thành thạo A) − P(thành thạo B)`, chỉ giữ cặp có ít nhất 50 người học thành thạo A và đã làm B. Chọn đúng 979 cạnh: mỗi khái niệm trước hết nhận cạnh vào có AV lớn nhất, phần còn lại lấy theo AV toàn cục. Đồ thị của phần k chỉ dùng người học thuộc tập huấn luyện của phần k.

**Đồ thị ngẫu nhiên.** 979 cạnh có hướng chọn ngẫu nhiên đều, đóng vai trò đối chứng âm: "chỉ cần có cấu trúc là đủ?".

**Khái niệm giữ lại.** Chọn ngẫu nhiên 15% khái niệm có tương tác (108 khái niệm, 257/979 cạnh chuyên gia chạm vào chúng). Không cạnh thống kê nào chạm vào các khái niệm này, nên trên đó đồ thị thống kê không thể khớp phân phối hành vi.

**Bảng I.** Đặc trưng của ba đồ thị (phần 0).

| Đồ thị | Cạnh | Nút cô lập | Thành phần liên thông yếu | Jaccard có hướng với chuyên gia |
|---|---|---|---|---|
| Chuyên gia | 979 | 59 | 69 | 1,000 |
| Thống kê | 979 | 331 | 332 | 0,010 |
| Ngẫu nhiên | 979 | 67 | 76 | 0,000 |

![Hình 2. Trái: phân phối bậc (vào + ra) của ba đồ thị. Phải: giá trị gia tăng của cạnh chuyên gia theo chiều xuôi, chiều ngược, của cạnh thống kê và của mọi cặp đủ hỗ trợ (tập huấn luyện phần 0).](report/figures/hinh2_do_thi.png)

Hai nguồn tri thức gần như tách rời: đồ thị thống kê chỉ trùng khoảng 1% cạnh với đồ thị chuyên gia (Bảng I). Trung vị AV của cạnh chuyên gia theo chiều xuôi (0,020) và chiều ngược (0,022) gần như bằng nhau và chỉ nhỉnh hơn mặt bằng (0,013): hành vi làm bài không phân biệt được chiều tiên quyết (Hình 2). Trong 331 nút cô lập của đồ thị thống kê, 221 nút buộc phải cô lập (108 khái niệm giữ lại và 113 khái niệm không có tương tác).

### 3.4. Huấn luyện

Mọi mô hình dùng Adam [@adam] với tốc độ học 10⁻³, entropy chéo nhị phân, cắt chuẩn gradient tại 1,0 và bỏ bước cập nhật khi chuẩn gradient không hữu hạn; cỡ lô 128 (AKT 64 do giới hạn bộ nhớ GPU T4). Huấn luyện tối đa 10 epoch, dừng sớm sau 3 epoch không cải thiện AUC kiểm định; mô hình tốt nhất được đánh giá trên tập kiểm tra. Giới hạn 10 epoch được chọn vì từ epoch 4, AUC kiểm định của GKT-Expert chỉ tăng khoảng 0,0005 mỗi epoch.

## 4. Thực nghiệm

### 4.1. Thiết lập

Kết quả là của phần 0: tập huấn luyện 122.799 người học (179.506 đoạn), tập kiểm định 30.700 và tập kiểm tra 38.375 người học. Độ đo chính là AUC-ROC; độ đo phụ là ACC (ngưỡng 0,5) và RMSE. AUC còn được tính riêng trên khái niệm giữ lại và theo nhóm khái niệm. Hiệu AUC giữa hai mô hình được kiểm định bằng phép DeLong ghép cặp [@delong; @fastdelong] trên cùng các dự đoán; phép kiểm định giả định các dự đoán độc lập nên với hàng triệu dự đoán, giá trị p rất nhỏ ngay cả khi hiệu nhỏ, và bài báo nhấn mạnh độ lớn của hiệu.

### 4.2. Kết quả chính

**Bảng II.** Kết quả trên tập kiểm tra (38.375 người học), phần 0; AUC cao nhất in đậm.

<!-- BANG1 -->

Thứ tự trên tập kiểm tra là DKT ({{auc:DKT}}) > GKT-Expert ({{auc:GKT-Expert}}) > GKT-Stat ({{auc:GKT-Stat}}) ≈ GKT-Random ({{auc:GKT-Random}}); ACC và RMSE cho cùng thứ tự (Bảng II). Ba điểm đáng chú ý. Thứ nhất, với cùng kiến trúc và cùng 979 cạnh, đồ thị chuyên gia hơn đồ thị thống kê {{delta:GKT-Expert:GKT-Stat}} và hơn đồ thị ngẫu nhiên {{delta:GKT-Expert:GKT-Random}}: tri thức sư phạm mang thông tin mà cấu trúc ngẫu nhiên không có. Thứ hai, đồ thị thống kê chỉ hơn đồ thị ngẫu nhiên {{delta:GKT-Stat:GKT-Random}}, tức các cạnh có giá trị gia tăng cao nhất gần như không giúp gì thêm cho lan truyền. Thứ ba, với RQ1, GKT-Expert vẫn thấp hơn DKT khoảng 0,005; trên Junyi, lan truyền theo đồ thị tiên quyết không vượt được mô hình tuần tự đơn giản, phù hợp với nhận định của simpleKT [@simplekt]. ⟨Bổ sung AKT khi chạy xong.⟩

![Hình 3. AUC kiểm định theo epoch (phần 0).](report/figures/hinh3_duong_cong_hoc.png)

### 4.3. Nghiên cứu cắt bỏ

**Bảng III.** Biến thể của GKT-Expert: xử lý nút cô lập và lan truyền một chiều.

<!-- BANG2 -->

⟨Nhận xét khi hai thí nghiệm cắt bỏ chạy xong.⟩

### 4.4. Khái niệm giữ lại và nhóm khái niệm

**Bảng IV.** AUC trên 108 khái niệm giữ lại và trên các khái niệm còn lại.

<!-- BANG3 -->

![Hình 4. AUC trên toàn tập kiểm tra và trên khái niệm giữ lại.](report/figures/hinh4_auc_giu_lai.png)

**Bảng V.** AUC theo nhóm khái niệm đích.

<!-- BANG4 -->

Mọi mô hình đều giảm khoảng 0,04 AUC trên khái niệm giữ lại, kể cả DKT không dùng đồ thị, nên mức giảm chủ yếu do các khái niệm này khó dự đoán hơn chứ không do đồ thị (Bảng IV). Điểm quyết định cho RQ3 là khoảng cách của GKT-Stat: {{auc_con_lai:GKT-Stat}} trên khái niệm còn lại và {{auc_giu_lai:GKT-Stat}} trên khái niệm giữ lại, gần như trùng với GKT-Random ({{auc_con_lai:GKT-Random}} và {{auc_giu_lai:GKT-Random}}). Đồ thị thống kê không tạo lợi thế ở nơi nó có cạnh, nên không có dấu hiệu rò rỉ gián tiếp làm thổi phồng kết quả. Ngược lại, GKT-Expert hơn GKT-Random {{delta:GKT-Expert:GKT-Random:khái niệm giữ lại}} trên khái niệm giữ lại, ngang với mức chênh trên toàn tập. Bảng V củng cố cơ chế: lợi thế của GKT-Expert xuất hiện ở khái niệm có cạnh chuyên gia (bậc thấp và bậc cao, khoảng 0,003) nhưng biến mất ở khái niệm cô lập trong đồ thị chuyên gia (0,8387 so với 0,8385), đúng như kỳ vọng vì nút cô lập không nhận thông điệp lân cận.

### 4.5. Kiểm định ghép cặp

**Bảng VI.** Kiểm định DeLong ghép cặp cho hiệu AUC (mô hình 1 − mô hình 2).

<!-- DELONG -->

### 4.6. Case study

<!-- CASE -->

![Hình 5. Case study: mức thành thạo dự đoán sau từng bước cho các khái niệm trong chuỗi tiên quyết. Dấu o: trả lời đúng, x: trả lời sai trên khái niệm của hàng đó.](report/figures/hinh5_case_study.png)

Ở bước 24, người học làm đúng `power_rule`; GKT-Expert nâng mức thành thạo dự đoán của `quotient_rule` từ 0,33 lên 0,71 và của `lhopitals_rule` từ 0,14 lên 0,48, dù người học chưa làm hai bài này: bằng chứng được truyền theo cạnh tiên quyết (Hình 5). DKT cũng thay đổi dự đoán cho các khái niệm khác, nhưng qua vector ẩn chung và không theo cấu trúc tường minh. Trên đoạn chuỗi này, tỉ lệ dự đoán đúng bước kế tiếp của DKT (0,81) vẫn cao hơn GKT-Expert (0,74) và GKT-Stat (0,76): khả năng diễn giải của GKT không đi kèm độ chính xác cao hơn.

## 5. Thảo luận

**RQ1.** Trên phần 0, GKT-Expert không vượt DKT. Các nguyên nhân khả dĩ: trạng thái 32 chiều cho mỗi khái niệm nhỏ hơn nhiều so với vector ẩn 64 chiều dùng chung của DKT; mỗi bước GKT chỉ cập nhật khái niệm được trả lời và lân cận, trong khi DKT cập nhật toàn bộ trạng thái; đồ thị Junyi là đồ thị bài tập với nhiều lỗi nhỏ và 113 khái niệm chỉ có trong đồ thị.

**RQ2 và câu hỏi phản biện thứ nhất.** Với cùng kiến trúc và số cạnh, đồ thị chuyên gia hơn đồ thị thống kê khoảng 0,003 AUC, còn đồ thị thống kê không hơn đồ thị ngẫu nhiên. Trên Junyi, tình huống "đồ thị thống kê cho AUC cao hơn" không xảy ra. Ngay cả khi xảy ra, điều đó cũng không tự động phủ nhận tri thức sư phạm, vì đồ thị thống kê được suy từ chính hành vi mà tập đánh giá đo lường; phép thử cần thiết là so sánh trên khái niệm giữ lại như trong bài báo.

**RQ3 và câu hỏi phản biện thứ hai.** Khoảng cách giữa khái niệm còn lại và khái niệm giữ lại của GKT-Stat (0,0396) bằng của GKT-Random (0,0395), nên không có bằng chứng về rò rỉ gián tiếp. Thiết kế đánh giá công bằng gồm: cùng kiến trúc và số cạnh, đồ thị thống kê chỉ dựng từ tập huấn luyện, khái niệm giữ lại không có cạnh thống kê, đối chứng âm bằng đồ thị ngẫu nhiên, và kiểm định ghép cặp trên cùng dự đoán. Các hiệu AUC khoảng 0,003 đều có p < 10⁻⁴ (Bảng VI), nhưng kiểm định DeLong bỏ qua tương quan trong cùng người học, nên cần xác nhận bằng nhiều phần dữ liệu và hạt giống.

**Hạn chế.** Mới chạy một phần dữ liệu với một hạt giống nên chưa có độ lệch chuẩn; kiểm định DeLong bỏ qua tương quan trong cùng người học; đồ thị chỉ có một loại cạnh. Hướng phát triển gồm chạy đủ 5 phần với bootstrap theo người học, học cấu trúc đồ thị với đồ thị chuyên gia làm tiên nghiệm, đồ thị nhiều loại cạnh theo SKT [@skt], và kết hợp cơ chế chú ý với lan truyền trên đồ thị.

## 6. Kết luận

Bài báo xây dựng GKT lan truyền hai chiều và một giao thức tách bạch đóng góp của tri thức chuyên gia khỏi tri thức thống kê trên dữ liệu Junyi. Đồ thị tiên quyết chuyên gia mang thông tin thật: hơn đồ thị thống kê và đồ thị ngẫu nhiên cùng số cạnh khoảng 0,003 AUC, kể cả trên khái niệm giữ lại. Đồ thị thống kê không hơn đồ thị ngẫu nhiên và không cho thấy dấu hiệu rò rỉ. Tuy vậy, lợi ích này nhỏ và không đủ để GKT vượt DKT; dữ liệu hành vi cũng không phân biệt được chiều tiên quyết. Mã nguồn và quy trình tái lập: https://github.com/Trilosphan/GraphK.

## Phụ lục

**Bảng VII.** Siêu tham số.

| Mô hình | Tham số chính | Cỡ lô | Tối ưu |
|---|---|---|---|
| GKT (ba đồ thị) | ẩn 32, nhúng 32, dropout 0,1, chuẩn hóa theo hàng | 128 | Adam 10⁻³, cắt gradient 1,0, tối đa 10 epoch |
| DKT | nhúng 64, LSTM 64, dropout 0,1 | 128 | như trên |
| AKT | `d_model` 256, 4 khối, 8 head, dropout 0,2 | 64 | như trên |
| Đồ thị thống kê | ngưỡng 0,5; hỗ trợ ≥ 50; 979 cạnh | — | — |

## Tài liệu tham khảo

<!-- TAI_LIEU -->
