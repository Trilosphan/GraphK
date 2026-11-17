# Đồ án 8 — GKT trên đồ thị tiên quyết tường minh

> Môn Machine Learning · Mô hình trọng tâm: GKT (Graph-based Knowledge Tracing)
> Dữ liệu chính: Junyi Academy (835 khái niệm, đồ thị tiên quyết do chuyên gia xác nhận) · Dữ liệu bổ sung: MOOCCubeX
> Mức độ: Chuyên sâu

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

Mỗi paper trình bày theo 5 ý: **phương pháp/giải thuật · dữ liệu · ưu · khuyết · kết quả đánh giá**. Mục cuối tổng hợp và đề xuất.

### 2.1. BKT — Corbett & Anderson (1994)
*Knowledge tracing: Modeling the acquisition of procedural knowledge.* User Modeling and User-Adapted Interaction, 4(4), 253–278.
- **Phương pháp:** HMM hai trạng thái cho mỗi kỹ năng, với 4 tham số (biết sẵn, học, đoán mò, trượt).
- **Ưu:** diễn giải được, ít tham số.
- **Khuyết:** các kỹ năng độc lập nhau, không dùng quan hệ tiên quyết.

### 2.2. DKT — Piech et al. (2015)
*Deep Knowledge Tracing.* NeurIPS 2015.
- **Phương pháp:** RNN/LSTM trên chuỗi one-hot `(câu hỏi, đúng/sai)`.
- **Dữ liệu:** ASSISTments, Khan Academy.
- **Ưu:** học được phụ thuộc giữa kỹ năng mà không cần gán nhãn.
- **Khuyết:** trạng thái ẩn là một vector chung, khó diễn giải theo từng khái niệm, quan hệ giữa khái niệm chỉ có dạng ẩn.

### 2.3. DKVMN — Zhang et al. (2017)
*Dynamic Key-Value Memory Networks for Knowledge Tracing.* WWW 2017.
- **Phương pháp:** bộ nhớ khóa (khái niệm, cố định) và bộ nhớ giá trị (mức nắm vững, cập nhật), dùng cơ chế đọc/ghi theo trọng số chú ý.
- **Ưu:** trạng thái theo từng khái niệm tiềm ẩn.
- **Khuyết:** khái niệm tiềm ẩn không gắn với đồ thị thật.

### 2.4. AKT — Ghosh, Heffernan & Lan (2020) — *mô hình đối chứng chính*
*Context-Aware Attentive Knowledge Tracing.* KDD 2020.
- **Phương pháp:** Transformer với chú ý suy giảm đơn điệu theo khoảng cách thời gian, embedding Rasch theo câu hỏi.
- **Ưu:** thuộc nhóm AUC cao nhất trên nhiều benchmark.
- **Khuyết:** không dùng cấu trúc tiên quyết.

### 2.5. GKT — Nakagawa, Iwasawa & Matsuo (2019) — *mô hình trọng tâm*
*Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.* IEEE/WIC/ACM WI 2019.
- **Phương pháp:**
  - Mỗi khái niệm có một trạng thái ẩn riêng.
  - Khi học sinh trả lời khái niệm `i`: tổng hợp đặc trưng của `i` và hàng xóm, cập nhật bằng `f_self` (cho `i`) và `f_neighbor` (cho hàng xóm), qua cổng erase–add, rồi GRU.
  - Dự đoán đúng/sai cho từng khái niệm.
- **Đồ thị trong bài gốc:**
  - Dựng sẵn: dense, transition (thống kê chuyển tiếp), DKT-based.
  - Học từ dữ liệu: PAM, MHA, VAE.
  - Bài gốc **không** dùng đồ thị chuyên gia. Đây chính là khoảng trống mà đồ án này lấp.
- **Dữ liệu:** ASSISTments 2009, KDD Cup 2010.
- **Ưu:** diễn giải được theo từng node, khai thác cấu trúc đồ thị.
- **Khuyết:** chi phí tính toán tăng theo số node; nhạy với chất lượng đồ thị.

### 2.6. (Mở rộng) SKT — Tong et al. (2020); GIKT — Yang et al. (2020)
- **SKT** (*Structure-based Knowledge Tracing: An Influence Propagation View*, ICDM 2020): lan truyền ảnh hưởng theo quan hệ tiên quyết (có hướng) và tương tự (vô hướng). Dùng để tham khảo cách tách loại cạnh (`edge_type`).
- **GIKT** (*GIKT: A Graph-based Interaction Model for Knowledge Tracing*, ECML-PKDD 2020): GCN trên đồ thị câu hỏi–kỹ năng.

### 2.7. Dữ liệu và benchmark
- **Junyi:** Chang, Hsu & Chen (2015), *Modeling Exercise Relationships in E-Learning: A Unified Approach*, EDM 2015. Đây là nguồn đồ thị tiên quyết.
- **MOOCCubeX:** Yu et al. (2021), CIKM 2021.
- **pyKT:** Liu et al. (2022), NeurIPS Datasets & Benchmarks. Có quy trình tiền xử lý chuẩn và cài đặt sẵn AKT, DKT, GKT để đối chứng.

### 2.8. Tổng hợp và đề xuất
- Bảng so sánh: phương pháp × dữ liệu × dùng đồ thị? (không / ẩn / thống kê / chuyên gia) × diễn giải × AUC báo cáo.
- **Khoảng trống:** chưa có đánh giá tách bạch giữa đồ thị chuyên gia và đồ thị thống kê trong cùng một kiến trúc GKT, với giao thức chống rò rỉ.
- **Đề xuất:** GKT dùng đồ thị chuyên gia Junyi, so với biến thể đồ thị thống kê và với AKT/DKT.

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

### 3.5. Pipeline mã nguồn
```
data/raw → src/preprocess.py → data/processed (chuỗi + node_id + A.npz)
         → src/graphs.py (expert / stat / random)
         → src/train.py --model {gkt,akt,dkt} --graph {...} --seed
         → results/*.json → src/report.py (bảng + biểu đồ)
```
- Cấu hình bằng YAML, cố định seed, log thí nghiệm, kiểm thử đơn vị cho bước dựng ma trận kề và chuẩn hóa.
- AKT/DKT: dùng cài đặt pyKT để bảo đảm đối chứng chuẩn.

---

## 4. Thử nghiệm

### 4.1. Dữ liệu
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
1. Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *UMUAI*, 4(4), 253–278.
2. Piech, C., et al. (2015). Deep Knowledge Tracing. *NeurIPS*.
3. Zhang, J., Shi, X., King, I., & Yeung, D.-Y. (2017). Dynamic Key-Value Memory Networks for Knowledge Tracing. *WWW*.
4. Nakagawa, H., Iwasawa, Y., & Matsuo, Y. (2019). Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network. *IEEE/WIC/ACM WI*.
5. Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. *KDD*.
6. Tong, S., et al. (2020). Structure-based Knowledge Tracing: An Influence Propagation View. *ICDM*.
7. Yang, Y., et al. (2020). GIKT: A Graph-based Interaction Model for Knowledge Tracing. *ECML-PKDD*.
8. Chang, H.-S., Hsu, H.-J., & Chen, K.-T. (2015). Modeling Exercise Relationships in E-Learning: A Unified Approach. *EDM*.
9. Yu, J., et al. (2021). MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs. *CIKM*.
10. Liu, Z., et al. (2022). pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models. *NeurIPS Datasets and Benchmarks*.
11. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
