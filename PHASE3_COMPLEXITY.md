## Phrase 3: Algorithm Complexity & Optimization

### 3.1. Phân tích Độ phức tạp Thời gian (Time Complexity)

Gọi các tham số:
- $N$ = Kích thước quần thể (Population size)
- $G$ = Số thế hệ tối đa (Generations)
- $n$ = Số thành phố (Chromosome length)
- $k$ = Kích thước Tournament

#### 3.1.1. Chi phí mỗi thế hệ

| Bước | Thao tác | Độ phức tạp |
|:-----|:---------|:------------|
| **Đánh giá Fitness** (`tsp_fitness`) | Duyệt toàn bộ $n$ gene trong chromosome | $O(N \cdot n)$ |
| **Sắp xếp quần thể** | Python `list.sort()` — Timsort | $O(N \log N)$ |
| **Tournament Selection** | Lấy mẫu $k$ phần tử, $N$ lần | $O(N \cdot k)$ |
| **Crossover (OX / PMX / CX)** | Cắt, fill, tra cứu trên chromosome | $O(N \cdot n)$ |
| **Mutation (Swap / Inversion / Scramble)** | Thao tác cục bộ trên slice | $O(N \cdot n)$ |

**Tổng độ phức tạp mỗi thế hệ:**

$$T_{\text{gen}} = O(N \cdot n + N \log N + N \cdot k) = O(N \cdot n)$$

> Với giả định $n \gg \log N$ và $n \gg k$ — điều này thường đúng trong TSP thực tế.

**Tổng độ phức tạp toàn bộ thuật toán:**

$$\boxed{T_{\text{GA}} = O(G \cdot N \cdot n)}$$

Với $G=500$, $N=100$, $n=52$ (berlin52): $\approx 2{,}600{,}000$ phép tính cơ bản — **rất khả thi trong thực tế.**

---

### 3.2. Phân tích Độ phức tạp Không gian (Space Complexity)

| Cấu trúc dữ liệu | Chi phí bộ nhớ |
|:-----------------|:---------------|
| Quần thể (`population`) — $N$ cá thể, mỗi cá thể $n$ gene | $O(N \cdot n)$ |
| Ma trận khoảng cách (`distance_matrix`) | $O(n^2)$ |
| Biến tạm (chromosome con, mảng tạm crossover) | $O(n)$ |

**Tổng:** $O(N \cdot n + n^2)$

---

### 3.3. Phân tích Chi tiết từng Toán tử

#### Crossover

| Toán tử | Cơ chế | Độ phức tạp | Điểm mạnh TSP |
|:--------|:-------|:------------|:--------------|
| **OX** (Ordered Crossover) | Copy đoạn P1 → fill tuần tự từ P2 (bỏ trùng) | $O(n)$ | Bảo tồn **thứ tự tương đối** — phù hợp nhất với TSP Euclidean |
| **PMX** (Partially Mapped) | Copy đoạn P1 → ánh xạ ngược P2 bằng dict | $O(n)$ avg | Bảo tồn **vị trí tuyệt đối** — phù hợp khi vị trí quan trọng |
| **CX** (Cycle Crossover) | Theo dõi vòng tròn (cycle) từ P1, P2 | $O(n)$ avg | Bảo tồn **vị trí tuyệt đối** của từng gene cùng lúc |

> **Lưu ý triển khai PMX:** Sử dụng dict `p2_pos = {v: k for k, v in enumerate(parent2)}` để tra cứu vị trí trong $O(1)$, giữ tổng phức tạp ở $O(n)$.

#### Mutation

| Toán tử | Cơ chế | Độ phức tạp | Tác động |
|:--------|:-------|:------------|:---------|
| **Swap** | Đổi chỗ 2 gene ngẫu nhiên | $O(1)$ | Thay đổi nhỏ, khám phá cục bộ |
| **Inversion** | Đảo ngược một đoạn con | $O(n)$ worst | Tương đương **2-opt** — loại giao điểm đường đi |
| **Scramble** | Xáo trộn một đoạn con ngẫu nhiên | $O(n)$ avg | Phá vỡ cấu trúc cục bộ mạnh hơn, tăng đa dạng |

---

### 3.4. Cơ chế Tối ưu đã Triển khai

#### A. Elitism
```python
new_population = self.population[:self.elite_size]
```
- **Tác dụng:** Đảm bảo cá thể tốt nhất không bao giờ bị mất (**monotone non-decreasing fitness**), giảm phương sai kết quả qua các lần chạy.
- **Độ phức tạp thêm:** $O(\text{elite\_size})$ — không đáng kể.

#### B. Tournament Selection (thay vì Roulette Wheel)
```python
def selection(self):
    sample = random.sample(self.population, 3)
    return max(sample, key=lambda x: x.fitness)
```
- **Ưu điểm:** Tạo áp lực chọn lọc có kiểm soát qua tham số $k$, **không bị chi phối** bởi cá thể siêu trội (dominant individual).
- **Tránh:** Hội tụ sớm (Premature Convergence) — vấn đề phổ biến của Roulette Wheel khi có cá thể có fitness vượt trội.

#### C. Hàm Fitness: Nghịch đảo khoảng cách
```python
return 1 / total_distance
```
- $f = 1/d_{\text{total}}$ → bài toán **tối thiểu hóa** khoảng cách được chuyển thành **tối đa hóa** fitness.
- Tương thích trực tiếp với cơ chế `sort(reverse=True)` và Tournament Selection.

---

### 3.5. So sánh với các Phương pháp khác trên TSP

| Thuật toán | Độ phức tạp | Chất lượng lời giải | Ghi chú |
|:-----------|:------------|:--------------------|:--------|
| **Brute Force** | $O(n!)$ | Tối ưu tuyệt đối | Chỉ áp dụng $n \leq 12$ |
| **Dynamic Programming (Held-Karp)** | $O(2^n \cdot n^2)$ | Tối ưu tuyệt đối | Áp dụng $n \leq 20$ |
| **Greedy (Nearest Neighbor)** | $O(n^2)$ | ~20–25% trên optimal | Nhanh, chất lượng thấp |
| **2-opt Local Search** | $O(n^2)$ / iteration | Tốt hơn Greedy | Dễ kẹt local optimum |
| **GA (cài đặt này)** | $O(G \cdot N \cdot n)$ | **~1–5% trên optimal** | Cân bằng tốt exploration/exploitation |

> **Kết luận Phrase 3:** GA không đảm bảo tối ưu tuyệt đối nhưng cung cấp lời giải **chất lượng cao trong thời gian đa thức** — đặc biệt phù hợp với TSP quy mô từ 50–200 thành phố nơi các phương pháp chính xác không khả thi.
