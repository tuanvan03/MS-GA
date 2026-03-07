# Thiết kế và Triển khai Thuật toán Genetic Algorithm (Hybrid GA)

Tài liệu này mô tả chi tiết thiết kế và triển khai thuật toán Genetic Algorithm (GA) kết hợp với tìm kiếm cục bộ (Local Search - 2-opt) để giải quyết bài toán Người du lịch (Traveling Salesman Problem - TSP), dựa trên mã nguồn thực tế trong file `GA.ipynb`.

## 1. Tổng quan Luồng hoạt động (High-Level Architecture)

Thuật toán hoạt động theo nguyên lý tiến hóa tự nhiên, với sự bổ sung của kỹ thuật Memetic (tìm kiếm cục bộ) để tinh chỉnh nghiệm. Quy trình tổng quát bao gồm các bước sau:

1.  **Khởi tạo (Initialization):** Tạo ra một quần thể ban đầu gồm $N$ cá thể ngẫu nhiên.
2.  **Đánh giá (Evaluation):** Tính độ thích nghi (Fitness) cho từng cá thể dựa trên độ dài lộ trình.
3.  **Vòng lặp Tiến hóa (Evolutionary Loop):** Lặp lại qua $G$ thế hệ:
    *   **Chọn lọc (Selection):** Chọn các cá thể tốt làm cha mẹ cho thế hệ sau (sử dụng Tournament Selection).
    *   **Lai ghép (Crossover):** Kết hợp gen của cha mẹ để tạo con cái (sử dụng toán tử OX).
    *   **Đột biến (Mutation):** Thay đổi ngẫu nhiên gen của con cái để duy trì sự đa dạng (sử dụng toán tử Inversion).
    *   **Tái tạo quần thể (Repopulation):** Tạo quần thể mới, áp dụng chiến lược Tinh hoa (Elitism) để giữ lại cá thể tốt nhất.
    *   **Tìm kiếm cục bộ (Local Search - Hybrid):** Áp dụng thuật toán 2-opt lên cá thể tốt nhất (Elite) định kỳ để tinh chỉnh nghiệm.
4.  **Kết thúc:** Trả về lộ trình tốt nhất tìm được sau khi hết số thế hệ quy định.

---

## 2. Giả mã Thuật toán (Pseudocode)

Dưới đây là giả mã chi tiết cho toàn bộ chương trình Hybrid GA.

### 2.1. Cấu trúc dữ liệu

*   **Individual (Cá thể):**
    *   `chromosome`: Mảng 1 chiều chứa hoán vị các thành phố (ví dụ: `[0, 5, 2, ..., 0]`).
    *   `fitness`: Giá trị độ thích nghi ($1 / \text{TotalDistance}$).
*   **Population (Quần thể):** Danh sách chứa `pop_size` cá thể.

### 2.2. Thuật toán Chính (Main Loop)

```python
Alogrithm: Hybrid_Genetic_Algorithm
Input: 
    DistanceMatrix D
    Parameters: PopSize, MutationRate, Generations, EliteSize, TournamentK
Output: 
    BestRoute, MinDistance

Begin
    1. Population P <- CreateInitialPopulation(PopSize)
    2. EvaluateFitness(P)
    
    3. For generation = 1 to Generations do:
        # a. Sắp xếp và Tinh hoa
        Sort P by Fitness descending
        NewPopulation P_new <- EmptyList
        
        # Giữ lại Top EliteSize cá thể tốt nhất
        Add P[0...EliteSize-1] to P_new
        
        # b. Lai ghép và Đột biến để điền đầy quần thể mới
        While Size(P_new) < PopSize do:
            Parent1 <- TournamentSelection(P, K=TournamentK)
            Parent2 <- TournamentSelection(P, K=TournamentK)
            
            Child_Chromosome <- Crossover_OX(Parent1, Parent2)
            Child_Chromosome <- Mutation_Inversion(Child_Chromosome, MutationRate)
            
            Add CreateIndividual(Child_Chromosome) to P_new
        End While
        
        P <- P_new
        EvaluateFitness(P)
        Sort P by Fitness descending
        
        # c. Hybrid Local Search (Memetic Step)
        # Áp dụng 2-opt định kỳ mỗi 50 thế hệ hoặc ở thế hệ cuối
        If (generation % 50 == 0) OR (generation == Generations) then:
            BestInd <- P[0]
            OptimizedRoute <- TwoOpt_LocalSearch(BestInd.chromosome, D)
            
            If Cost(OptimizedRoute) < Cost(BestInd.chromosome) then:
                BestInd.chromosome <- OptimizedRoute
                UpdateFitness(BestInd)
            End If
        End If
        
        # Lưu lịch sử hội tụ
        RecordBestCost(P[0])
    End For

    Return P[0].chromosome, 1/P[0].fitness
End
```

### 2.3. Chi tiết Các Toán tử (Operators)

#### A. Evaluate Fitness
Hàm mục tiêu là tối thiểu hóa khoảng cách, nên hàm thích nghi là nghịch đảo của tổng khoảng cách.
```python
Function Fitness(Chromosome C):
    TotalDistance <- 0
    For i from 0 to Length(C)-1 do:
        u <- C[i]
        v <- C[(i+1) % Length(C)]
        TotalDistance <- TotalDistance + D[u][v]
    Return 1.0 / TotalDistance
```

#### B. Tournament Selection
Chọn `k` cá thể ngẫu nhiên từ quần thể và lấy cá thể tốt nhất trong nhóm đó. Giúp duy trì áp lực chọn lọc nhưng tránh hội tụ quá sớm.
```python
Function TournamentSelection(Population P, K):
    Sample <- RandomSelect(P, K)
    Return Individual with Max Fitness in Sample
```

#### C. Order Crossover (OX)
Toán tử lai ghép phù hợp cho bài toán hoán vị, bảo tồn thứ tự tương đối của các gene.
```python
Function Crossover_OX(Parent1, Parent2):
    Size <- Length(Parent1)
    Child <- Array of Size (Empty)
    
    # 1. Cắt một đoạn ngẫu nhiên từ Parent1
    idx1, idx2 <- RandomRange(0, Size)
    Child[idx1:idx2] <- Parent1[idx1:idx2]
    
    # 2. Điền các gene còn thiếu từ Parent2 theo thứ tự xuất hiện
    P2_Pointer <- (idx2 + 1) % Size
    Child_Pointer <- (idx2 + 1) % Size
    
    While Child is not full do:
        Gene <- Parent2[P2_Pointer]
        If Gene is NOT in Child then:
            Child[Child_Pointer] <- Gene
            Child_Pointer <- (Child_Pointer + 1) % Size
        End If
        P2_Pointer <- (P2_Pointer + 1) % Size
    End While
    
    Return Child
```

#### D. Inversion Mutation
Đảo ngược một đoạn gene ngẫu nhiên. Tương đương với phép biến đổi 2-opt ngẫu nhiên, giúp gỡ các nút thắt chéo trong lộ trình.
```python
Function Mutation_Inversion(Chromosome C, Rate):
    If Random(0, 1) < Rate then:
        i, j <- RandomRange(0, Length(C))
        Reverse sub-segment C[i:j]
    Return C
```

#### E. 2-opt Local Search
Duyệt qua tất cả các cặp cạnh không kề nhau và thử đảo ngược đường đi để loại bỏ giao cắt.
```python
Function TwoOpt_LocalSearch(Route, DistMatrix):
    Improved <- True
    BestDist <- Cost(Route)
    
    While Improved is True do:
        Improved <- False
        For i from 1 to N-2 do:
            For j from i+1 to N-1 do:
                # Tính Delta nếu đảo ngược đoạn từ i đến j
                OldDist <- D[Route[i-1]][Route[i]] + D[Route[j]][Route[j+1]]
                NewDist <- D[Route[i-1]][Route[j]] + D[Route[i]][Route[j+1]]
                
                If NewDist < OldDist then:
                    Reverse Route[i...j]
                    Improved <- True
                End If
            End For
        End For
    End While
    Return Route
```

---

## 3. Cấu hình Tham số Thực nghiệm (Parameters)

Dựa trên kết quả thực nghiệm trong `GA.ipynb`, bộ tham số tối ưu được lựa chọn như sau:

| Tham số | Giá trị | Giải thích |
| :--- | :--- | :--- |
| **Population Size** | 100 | Cân bằng giữa đa dạng gen và tốc độ tính toán. |
| **Generations** | 500 - 1000 | Đủ lớn để thuật toán hội tụ trên các dataset < 100 cities. |
| **Mutation Rate** | 0.25 | Xác suất đột biến khá cao để duy trì sự đa dạng. |
| **Crossover Type** | OX (Order Crossover) | Tốt nhất cho việc bảo tồn thứ tự đường đi. |
| **Mutation Type** | Inversion | Hiệu quả nhất trong việc gỡ các nút thắt hình học. |
| **Selection** | Tournament (k=5) | Áp lực chọn lọc vừa phải. |
| **Elitism** | 1 | Luôn giữ lại cá thể tốt nhất để đảm bảo kết quả không bị suy thoái. |
| **Hybrid Frequency** | Mỗi 50 thế hệ | Tần suất chạy 2-opt để tinh chỉnh mà không tốn quá nhiều chi phí tính toán. |

---

## 4. Kết luận
Thiết kế này kết hợp sức mạnh tìm kiếm toàn cục (Global Search) của GA để khám phá không gian nghiệm rộng lớn và sức mạnh tìm kiếm cục bộ (Local Search) của 2-opt để khai thác cực trị địa phương. Kết quả thực nghiệm cho thấy phương pháp lai (Hybrid) này vượt trội hơn hẳn so với GA thuần túy, đặc biệt là trên các bộ dữ liệu lớn như `kroA100`.
