# ĐỀ CƯƠNG THỰC NGHIỆM: ĐÁNH GIÁ HIỆU NĂNG THUẬT TOÁN DI TRUYỀN (GA) TRÊN BÀI TOÁN TSP

## 1. Mục tiêu và Giả thuyết Nghiên cứu (Research Objectives & Hypotheses)

Mục tiêu của thực nghiệm là xác định tổ hợp các toán tử di truyền (Genetic Operators) tối ưu nhất để giải quyết bài toán Người du lịch (TSP), dựa trên sự cân bằng giữa chất lượng lời giải và tốc độ hội tụ.

*   **Giả thuyết H1 (Hiệu quả của phép lai - Crossover):** Toán tử **Ordered Crossover (OX)** sẽ mang lại chất lượng lời giải tốt hơn so với PMX và CX do khả năng bảo tồn thứ tự tương đối của các thành phố từ cha mẹ, giảm thiểu việc phá vỡ các chuỗi gen tốt (building blocks).
*   **Giả thuyết H2 (Hiệu quả của phép đột biến - Mutation):** Toán tử **Inversion Mutation** sẽ giúp thuật toán hội tụ nhanh hơn và thoát khỏi cực trị địa phương tốt hơn Swap và Scramble, nhờ cơ chế đảo ngược tương tự kỹ thuật tối ưu cục bộ *2-opt* (giúp loại bỏ các đường chéo cắt nhau).
*   **Giả thuyết H3 (Tính ổn định - Robustness):** Tổ hợp **[OX + Inversion]** sẽ có độ lệch chuẩn (Standard Deviation) thấp nhất qua 30 lần chạy, khẳng định tính ổn định cao nhất của cấu hình này.

## 2. Thiết lập Môi trường và Tham số (Experimental Setup)

Để đảm bảo tính công bằng và độ tin cậy thống kê, các tham số nền tảng sẽ được cố định như sau:

| Tham số (Parameter) | Giá trị (Value) | Giải thích (Rationale) |
| :--- | :--- | :--- |
| **Kích thước quần thể ($N$)** | $100$ | Cân bằng giữa sự đa dạng di truyền và chi phí tính toán. |
| **Số thế hệ tối đa ($G$)** | $500$ | Điểm bão hòa dự kiến đối với các instance $<100$ thành phố. |
| **Phương pháp Chọn lọc** | Tournament (k=5) | *[Bổ sung]* Tạo áp lực chọn lọc tốt hơn Roulette Wheel, tránh hội tụ sớm. |
| **Tỷ lệ Lai ghép ($P_c$)** | $0.9$ | *[Bổ sung]* Đảm bảo khai thác tối đa thông tin từ thế hệ cha mẹ. |
| **Tỷ lệ Đột biến ($P_m$)** | $0.05$ | Giữ sự đa dạng cho quần thể, tránh mắc kẹt tại cực trị địa phương. |
| **Cơ chế Elitism** | Giữ 1 cá thể tốt nhất | *[Bổ sung]* Đảm bảo Fitness tốt nhất không bao giờ bị suy giảm (monotonicity). |
| **Số lần chạy độc lập** | $30$ | Đảm bảo ý nghĩa thống kê (Statistical Significance). |
| **Dataset chuẩn** | `berlin52` (TSPLIB) | Optima đã biết: 7542. Dùng để tính sai số tương đối (Gap). |

## 3. Quy trình Thực nghiệm (Experimental Procedure)

### Giai đoạn 1: Đánh giá so sánh toán tử (Operator Comparative Analysis)
Thực hiện chiến lược *Full Factorial Design* với ma trận $3 \times 3$ để tìm ra cấu hình tối ưu.
*   **Biến thiên Lai ghép (3):** OX (Ordered), PMX (Partially Mapped), CX (Cycle).
*   **Biến thiên Đột biến (3):** Swap (Đổi chỗ), Inversion (Đảo ngược), Scramble (Trộn lẫn).
*   **Tổng số thí nghiệm:** 9 cấu hình $\times$ 30 lần chạy = 270 lần thực thi.
*   **Chỉ số đo lường:** Best Fitness (tốt nhất), Mean Fitness (trung bình), Std Dev (độ lệch), Thời gian tính toán.

### Giai đoạn 2: Phân tích độ nhạy tham số (Sensitivity Analysis)
Lựa chọn cấu hình tốt nhất từ Giai đoạn 1 (Dự kiến: OX + Inversion) để kiểm tra độ nhạy với tỷ lệ đột biến ($P_m$).
*   **Các mức thử nghiệm:**
    *   $P_m = 0.01$ (Thấp): Kiểm tra rủi ro hội tụ sớm (Premature Convergence).
    *   $P_m = 0.05$ (Trung bình): Mức tiêu chuẩn.
    *   $P_m = 0.20$ (Cao): Kiểm tra sự phá vỡ cấu trúc gen (Random Walk behavior).

### Giai đoạn 3: Kiểm chứng khả năng mở rộng (Scalability Test)
Đánh giá hiệu năng của cấu hình tối ưu trên các quy mô bài toán khác nhau từ thư viện TSPLIB:
1.  **Quy mô nhỏ:** `burma14` (14 thành phố) - Kiểm tra khả năng tìm ra Global Optimum tuyệt đối.
2.  **Quy mô vừa:** `berlin52` (52 thành phố) - Benchmark chính.
3.  **Quy mô lớn:** `kroA100` hoặc `ch130` (>100 thành phố) - Đánh giá sự suy giảm hiệu năng khi không gian tìm kiếm tăng theo hàm giai thừa.

## 4. Phương pháp Đánh giá & Trực quan hóa (Evaluation & Visualization)

Báo cáo kết quả sẽ bao gồm các biểu đồ phân tích sâu:

1.  **Biểu đồ Hội tụ (Convergence Plot):**
    *   Đường biểu diễn Fitness trung bình qua các thế hệ của 3 nhóm lai ghép (OX, PMX, CX).
    *   *Mục đích:* So sánh tốc độ hội tụ (Convergence Speed).

2.  **Biểu đồ Hộp (Boxplot):**
    *   Hiển thị phân phối kết quả của 9 tổ hợp sau 30 lần chạy.
    *   *Mục đích:* Đánh giá độ ổn định và phát hiện các giá trị ngoại lai (outliers).

3.  **Trực quan hóa Lộ trình (Route Visualization):**
    *   So sánh trực quan lộ trình tại Gen 1 vs. Gen 500.
    *   *Mục đích:* Minh họa khả năng gỡ rối các đường chéo cắt nhau của thuật toán.

4.  **Bảng tổng hợp chỉ số (Performance Table):**
    *   Bao gồm: Best Cost, Mean Cost, Std Dev, và **% Gap** (Sai số so với kết quả tối ưu của TSPLIB).
    *   Công thức Gap: $Gap (\%) = \frac{Best_{GA} - Opt_{TSPLIB}}{Opt_{TSPLIB}} \times 100$

## 5. Kết luận Dự kiến
Dựa trên lý thuyết về Heuristic, kết quả thực nghiệm kỳ vọng sẽ chứng minh việc kết hợp khả năng giữ trật tự gen của **OX** và khả năng tối ưu cục bộ hình học của **Inversion** là chiến lược hiệu quả nhất cho TSP dạng Euclide 2D.