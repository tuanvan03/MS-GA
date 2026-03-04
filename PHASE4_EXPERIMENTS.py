"""
============================================================
PHASE 4: EXPERIMENTS — Genetic Algorithm for TSP
Dựa trên: EXPERIMENT.md
Dataset: TSPLIB (burma14, berlin52, kroA100)
         Các file .tsp phải nằm trong thư mục ./data/
============================================================
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from random import sample as rand_sample

# ============================================================
# 0. IMPORT CÁC CLASS/HÀM TỪ GA.ipynb (copy hoặc import)
# ============================================================

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0


class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, elite_size,
                 crossover_func, mutate_func, fitness_func, tournament_k=5):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.fitness_func = fitness_func
        self.tournament_k = tournament_k
        self.population = []

    def create_initial_population(self, create_individual_func):
        self.population = [Individual(create_individual_func()) for _ in range(self.pop_size)]

    def evolve(self):
        # 1. Evaluate fitness
        for ind in self.population:
            ind.fitness = self.fitness_func(ind.chromosome)
        # 2. Sort
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        # 3. Elitism + fill
        new_population = [deepcopy(ind) for ind in self.population[:self.elite_size]]
        while len(new_population) < self.pop_size:
            p1 = self.selection()
            p2 = self.selection()
            child_chrom = self.crossover_func(p1.chromosome[:], p2.chromosome[:])
            child_chrom = self.mutate_func(child_chrom)
            new_population.append(Individual(child_chrom))
        self.population = new_population

    def selection(self):
        """Tournament Selection với k phần tử."""
        candidates = random.sample(self.population, self.tournament_k)
        return max(candidates, key=lambda x: x.fitness)


# ============================================================
# 1. CÁC TOÁN TỬ TSP
# ============================================================

def tsp_crossover_ox(parent1, parent2):
    """Ordered Crossover (OX)."""
    size = len(parent1)
    idx1, idx2 = sorted(rand_sample(range(size), 2))
    child = [None] * size
    child[idx1:idx2+1] = parent1[idx1:idx2+1]
    genes_in_child = set(child[idx1:idx2+1])
    p2_ptr = (idx2 + 1) % size
    c_ptr  = (idx2 + 1) % size
    added = 0
    total_fill = size - (idx2 - idx1 + 1)
    while added < total_fill:
        gene = parent2[p2_ptr]
        if gene not in genes_in_child:
            child[c_ptr] = gene
            c_ptr = (c_ptr + 1) % size
            added += 1
        p2_ptr = (p2_ptr + 1) % size
    return child


def tsp_crossover_pmx(parent1, parent2):
    """Partially Mapped Crossover (PMX)."""
    size = len(parent1)
    child = [None] * size
    idx1, idx2 = sorted(rand_sample(range(size), 2))
    child[idx1:idx2+1] = parent1[idx1:idx2+1]
    p2_pos = {v: k for k, v in enumerate(parent2)}
    for i in range(idx1, idx2+1):
        gene_p2 = parent2[i]
        if gene_p2 not in child[idx1:idx2+1]:
            target = p2_pos[parent1[i]]
            while idx1 <= target <= idx2:
                target = p2_pos[parent1[target]]
            child[target] = gene_p2
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child


def tsp_crossover_cx(parent1, parent2):
    """Cycle Crossover (CX)."""
    size = len(parent1)
    child = [None] * size
    idx = 0
    while child[idx] is None:
        child[idx] = parent1[idx]
        val_p2 = parent2[idx]
        idx = parent1.index(val_p2)
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child


CROSSOVER_FUNCS = {
    'ox' : tsp_crossover_ox,
    'pmx': tsp_crossover_pmx,
    'cx' : tsp_crossover_cx,
}


def make_mutate_func(mutation_type, mutation_rate):
    """Factory trả về hàm mutation với type và rate cố định."""
    def mutate(chromosome):
        if random.random() < mutation_rate:
            c = chromosome[:]
            if mutation_type == 'swap':
                i, j = rand_sample(range(len(c)), 2)
                c[i], c[j] = c[j], c[i]
            elif mutation_type == 'inversion':
                i, j = sorted(rand_sample(range(len(c)), 2))
                c[i:j+1] = c[i:j+1][::-1]
            elif mutation_type == 'scramble':
                i, j = sorted(rand_sample(range(len(c)), 2))
                sub = c[i:j+1]
                random.shuffle(sub)
                c[i:j+1] = sub
            return c
        return chromosome
    return mutate


def make_fitness_func(dist_matrix):
    """Factory trả về hàm fitness đóng gói dist_matrix."""
    def fitness(chromosome):
        total = 0.0
        n = len(chromosome)
        for i in range(n):
            total += dist_matrix[chromosome[i]][chromosome[(i+1) % n]]
        return 1.0 / total
    return fitness


# ============================================================
# 2. TIỆN ÍCH: Load dữ liệu TSPLIB & build ma trận khoảng cách
# ============================================================

def load_tsplib(filepath):
    """Đọc file .tsp định dạng TSPLIB, trả về list tọa độ [(x, y), ...]."""
    coords = []
    with open(filepath, 'r') as f:
        reading = False
        for line in f:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                reading = True
                continue
            if line in ('EOF', ''):
                if reading:
                    break
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords


def build_distance_matrix(coords):
    """Xây dựng ma trận khoảng cách Euclidean nguyên (nint) — chuẩn TSPLIB."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = int(np.sqrt(dx**2 + dy**2) + 0.5)  # nint
    return dist

@Ph
# ============================================================
# 3. HÀM CHẠY MỘT THÍ NGHIỆM ĐƠN (Single Run)
# ============================================================

def run_single_experiment(n_cities, dist_matrix,
                           crossover_type='ox', mutation_type='inversion',
                           pop_size=100, generations=500,
                           mutation_rate=0.05, elite_size=1,
                           tournament_k=5, seed=None):
    """
    Chạy một lần GA với cấu hình cho trước.

    Returns:
        best_cost (float): Khoảng cách tốt nhất tìm được.
        history   (list) : Best cost tại mỗi thế hệ.
        best_chromosome (list): Chromosome tốt nhất.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fitness_func   = make_fitness_func(dist_matrix)
    crossover_func = CROSSOVER_FUNCS[crossover_type]
    mutate_func    = make_mutate_func(mutation_type, mutation_rate)

    ga = GeneticAlgorithm(
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        crossover_func=crossover_func,
        mutate_func=mutate_func,
        fitness_func=fitness_func,
        tournament_k=tournament_k,
    )
    ga.create_initial_population(lambda: random.sample(range(n_cities), n_cities))

    history = []
    for _ in range(generations):
        ga.evolve()
        # Cập nhật fitness sau evolve (vì new_population chưa được evaluate)
        for ind in ga.population:
            if ind.fitness == 0.0:
                ind.fitness = fitness_func(ind.chromosome)
        ga.population.sort(key=lambda x: x.fitness, reverse=True)
        history.append(round(1.0 / ga.population[0].fitness, 2))

    best_cost       = history[-1]
    best_chromosome = ga.population[0].chromosome[:]
    return best_cost, history, best_chromosome


# ============================================================
# 4. LOAD DATASETS
# ============================================================

DATASETS = {
    'burma14' : {'path': 'data/burma14.tsp',  'optimum': 3323},
    'berlin52': {'path': 'data/berlin52.tsp',  'optimum': 7542},
    'kroA100' : {'path': 'data/kroA100.tsp',   'optimum': 21282},
}

print("Loading datasets...")
loaded = {}
for name, info in DATASETS.items():
    try:
        coords = load_tsplib(info['path'])
        dist   = build_distance_matrix(coords)
        loaded[name] = {'coords': coords, 'dist': dist, 'n': len(coords), 'optimum': info['optimum']}
        print(f"  ✅ {name}: {len(coords)} thành phố | Known optimum: {info['optimum']}")
    except FileNotFoundError:
        print(f"  ⚠️  {name}: File không tìm thấy tại '{info['path']}'. Bỏ qua.")

# Benchmark chính
BENCHMARK = 'berlin52'
assert BENCHMARK in loaded, f"Dataset {BENCHMARK} chưa được load!"
coords_main = loaded[BENCHMARK]['coords']
dist_main   = loaded[BENCHMARK]['dist']
N_CITIES    = loaded[BENCHMARK]['n']
OPTIMUM     = loaded[BENCHMARK]['optimum']
print(f"\nBenchmark chính: {BENCHMARK} ({N_CITIES} thành phố, optimum={OPTIMUM})")


# ============================================================
# GIAI ĐOẠN 1: Full Factorial Design 3×3
# 9 cấu hình × 30 lần chạy = 270 lần thực thi
# Tham số cố định: N=100, G=500, Pm=0.05, k=5, elitism=1
# ============================================================

CROSSOVER_TYPES = ['ox', 'pmx', 'cx']
MUTATION_TYPES  = ['swap', 'inversion', 'scramble']
N_RUNS          = 30

results_phase1  = {}
histories_phase1 = {}

print("\n" + "=" * 65)
print("GIAI ĐOẠN 1: Full Factorial Design 3×3 — berlin52")
print(f"Tổng số lần chạy: {len(CROSSOVER_TYPES) * len(MUTATION_TYPES) * N_RUNS}")
print("=" * 65)

for cx_type in CROSSOVER_TYPES:
    for mut_type in MUTATION_TYPES:
        key    = (cx_type, mut_type)
        costs  = []
        hists  = []
        t0     = time.time()

        for run in range(N_RUNS):
            cost, hist, _ = run_single_experiment(
                n_cities=N_CITIES, dist_matrix=dist_main,
                crossover_type=cx_type, mutation_type=mut_type,
                pop_size=100, generations=500,
                mutation_rate=0.05, elite_size=1,
                tournament_k=5, seed=run
            )
            costs.append(cost)
            hists.append(hist)

        elapsed = time.time() - t0
        results_phase1[key]   = costs
        histories_phase1[key] = np.mean(hists, axis=0).tolist()

        best = min(costs)
        mean = np.mean(costs)
        std  = np.std(costs)
        gap  = (best - OPTIMUM) / OPTIMUM * 100
        print(f"[{cx_type.upper():3s}+{mut_type:10s}] "
              f"Best={best:7.1f} | Mean={mean:7.1f} | Std={std:6.1f} | "
              f"Gap={gap:+.2f}% | {elapsed:5.1f}s")

print("=" * 65)

# ---- Trực quan hóa Giai đoạn 1 ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Giai đoạn 1: So sánh 9 cấu hình toán tử GA (berlin52)',
             fontsize=14, fontweight='bold')

colors_cx  = {'ox': '#2980B9', 'pmx': '#27AE60', 'cx': '#E74C3C'}
styles_mut = {'swap': '-', 'inversion': '--', 'scramble': ':'}

ax1 = axes[0]
for (cx, mut), hist in histories_phase1.items():
    ax1.plot(hist, color=colors_cx[cx], linestyle=styles_mut[mut],
             linewidth=1.5, label=f'{cx.upper()}+{mut}', alpha=0.85)
ax1.axhline(y=OPTIMUM, color='black', linestyle='-.', linewidth=1.5,
            label=f'Optimum ({OPTIMUM})')
ax1.set_xlabel('Thế hệ (Generation)')
ax1.set_ylabel('Best Cost (khoảng cách)')
ax1.set_title('Đường hội tụ trung bình (30 lần chạy)')
ax1.legend(fontsize=7.5, ncol=2)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
labels_box = [f'{cx.upper()}\n+{mut}' for (cx, mut) in results_phase1]
bp = ax2.boxplot(list(results_phase1.values()), labels=labels_box,
                 patch_artist=True, showfliers=True)
palette = ['#AED6F1','#A9DFBF','#F9E79F','#F5CBA7','#D2B4DE',
           '#A3E4D7','#FADBD8','#D5D8DC','#F0B27A']
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color)
ax2.axhline(y=OPTIMUM, color='red', linestyle='--', linewidth=1.2, label='Optimum')
ax2.set_ylabel('Best Cost')
ax2.set_title('Phân phối kết quả sau 30 lần chạy')
ax2.tick_params(axis='x', labelsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase1_operator_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: phase1_operator_comparison.png")

# ---- Bảng tổng hợp ----
rows = []
for (cx, mut), costs in results_phase1.items():
    best = min(costs)
    rows.append({
        'Crossover': cx.upper(), 'Mutation': mut,
        'Best Cost': round(best, 1), 'Mean Cost': round(np.mean(costs), 1),
        'Std Dev':   round(np.std(costs), 1),
        'Gap (%)':   round((best - OPTIMUM) / OPTIMUM * 100, 3),
    })
df1 = pd.DataFrame(rows).sort_values('Best Cost').reset_index(drop=True)
print("\nBẢNG KẾT QUẢ GIAI ĐOẠN 1:")
print(df1.to_string(index=False))

# Xác định cấu hình tốt nhất
best_row       = df1.iloc[0]
BEST_CROSSOVER = best_row['Crossover'].lower()
BEST_MUTATION  = best_row['Mutation']
print(f"\n>>> Cấu hình tốt nhất: [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]")


# ============================================================
# GIAI ĐOẠN 2: Phân tích Độ nhạy tham số Pm
# Cấu hình: best từ GĐ1 | Pm ∈ {0.01, 0.05, 0.20}
# ============================================================

PM_LEVELS  = [0.01, 0.05, 0.20]
PM_LABELS  = {0.01: 'Pm=0.01 (Thấp)', 0.05: 'Pm=0.05 (Trung bình)', 0.20: 'Pm=0.20 (Cao)'}
PM_COLORS  = {0.01: '#E74C3C', 0.05: '#2ECC71', 0.20: '#3498DB'}

results_phase2  = {}
histories_phase2 = {}

print("\n" + "=" * 65)
print(f"GIAI ĐOẠN 2: Độ nhạy Pm — [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]")
print("=" * 65)

for pm in PM_LEVELS:
    costs = []
    hists = []
    for run in range(N_RUNS):
        cost, hist, _ = run_single_experiment(
            n_cities=N_CITIES, dist_matrix=dist_main,
            crossover_type=BEST_CROSSOVER, mutation_type=BEST_MUTATION,
            pop_size=100, generations=500,
            mutation_rate=pm, elite_size=1, tournament_k=5, seed=run
        )
        costs.append(cost)
        hists.append(hist)

    results_phase2[pm]   = costs
    histories_phase2[pm] = np.mean(hists, axis=0).tolist()

    best = min(costs)
    gap  = (best - OPTIMUM) / OPTIMUM * 100
    print(f"{PM_LABELS[pm]:25s} | Best={best:7.1f} | "
          f"Mean={np.mean(costs):7.1f} | Std={np.std(costs):6.1f} | Gap={gap:+.2f}%")

print("=" * 65)

# ---- Trực quan hóa GĐ2 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Giai đoạn 2: Độ nhạy Pm — [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
for pm, hist in histories_phase2.items():
    ax1.plot(hist, color=PM_COLORS[pm], linewidth=2, label=PM_LABELS[pm])
ax1.axhline(y=OPTIMUM, color='black', linestyle='--', linewidth=1.5, label='Optimum')
ax1.set_xlabel('Thế hệ')
ax1.set_ylabel('Best Cost')
ax1.set_title('Đường hội tụ theo mức Pm')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
bp2 = ax2.boxplot(
    [results_phase2[pm] for pm in PM_LEVELS],
    labels=[PM_LABELS[pm] for pm in PM_LEVELS],
    patch_artist=True
)
for patch, pm in zip(bp2['boxes'], PM_LEVELS):
    patch.set_facecolor(PM_COLORS[pm])
    patch.set_alpha(0.75)
ax2.axhline(y=OPTIMUM, color='red', linestyle='--', linewidth=1.2)
ax2.set_ylabel('Best Cost')
ax2.set_title('Phân phối kết quả theo mức Pm')
ax2.tick_params(axis='x', labelsize=8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase2_sensitivity_pm.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: phase2_sensitivity_pm.png")


# ============================================================
# GIAI ĐOẠN 3: Scalability Test
# Dataset: burma14, berlin52, kroA100
# Cấu hình: best từ GĐ1, Pm=0.05
# ============================================================

results_phase3 = {}

print("\n" + "=" * 65)
print(f"GIAI ĐOẠN 3: Scalability Test — [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]")
print("=" * 65)

for ds_name, ds_info in loaded.items():
    n       = ds_info['n']
    dist_m  = ds_info['dist']
    opt     = ds_info['optimum']
    gens    = 300 if n <= 20 else 500   # Ít hơn cho dataset nhỏ

    costs = []
    hists = []
    t0 = time.time()
    for run in range(N_RUNS):
        cost, hist, _ = run_single_experiment(
            n_cities=n, dist_matrix=dist_m,
            crossover_type=BEST_CROSSOVER, mutation_type=BEST_MUTATION,
            pop_size=100, generations=gens,
            mutation_rate=0.05, elite_size=1, tournament_k=5, seed=run
        )
        costs.append(cost)
        hists.append(hist)

    elapsed = time.time() - t0
    best = min(costs)
    gap  = (best - opt) / opt * 100
    results_phase3[ds_name] = {
        'n': n, 'costs': costs, 'hists': hists, 'best': best,
        'mean': np.mean(costs), 'std': np.std(costs),
        'gap': gap, 'optimum': opt, 'time': elapsed,
        'avg_hist': np.mean(hists, axis=0).tolist()
    }
    print(f"{ds_name:10s} ({n:3d} TP) | "
          f"Best={best:8.1f} | Gap={gap:+.2f}% | "
          f"Std={np.std(costs):7.1f} | {elapsed:6.1f}s")

print("=" * 65)

# ---- Trực quan hóa GĐ3: Tọa độ thành phố + Route ----
n_plots = len(results_phase3)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
if n_plots == 1:
    axes = [axes]
fig.suptitle('Giai đoạn 3: Best Route trên từng dataset (TSPLIB)',
             fontsize=13, fontweight='bold')

bar_colors = ['#5DADE2', '#52BE80', '#F39C12']
for idx, (ds_name, info) in enumerate(results_phase3.items()):
    ax = axes[idx]
    coords  = loaded[ds_name]['coords']
    dist_m  = loaded[ds_name]['dist']
    n_c     = info['n']
    opt     = info['optimum']

    # Lấy best route bằng cách chạy lại với seed tốt nhất
    best_seed = int(np.argmin(info['costs']))
    _, _, best_route = run_single_experiment(
        n_cities=n_c, dist_matrix=dist_m,
        crossover_type=BEST_CROSSOVER, mutation_type=BEST_MUTATION,
        pop_size=100, generations=300 if n_c <= 20 else 500,
        mutation_rate=0.05, seed=best_seed
    )

    xs = [coords[c][0] for c in best_route] + [coords[best_route[0]][0]]
    ys = [coords[c][1] for c in best_route] + [coords[best_route[0]][1]]

    ax.plot(xs, ys, '-', color=bar_colors[idx], linewidth=1.0, alpha=0.7)
    ax.scatter([c[0] for c in coords], [c[1] for c in coords],
               s=35, color=bar_colors[idx], zorder=5)
    ax.scatter([coords[best_route[0]][0]], [coords[best_route[0]][1]],
               s=100, color='red', zorder=6, marker='*', label='Start')

    ax.set_title(
        f"{ds_name} ({n_c} cities)\n"
        f"Best={info['best']:.0f} | Gap={info['gap']:+.2f}% | Opt={opt}",
        fontsize=9
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('phase3_scalability_routes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: phase3_scalability_routes.png")

# ---- Biểu đồ Hội tụ & Boxplot GĐ3 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Giai đoạn 3: Hội tụ và Phân phối (Gap %) — [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
for idx, (ds_name, info) in enumerate(results_phase3.items()):
    opt = info['optimum']
    # Chuẩn hóa về % Gap để dễ dàng hiển thị chung 1 trục y do chênh lệch scale lớn
    gap_hist = [(val - opt) / opt * 100 for val in info['avg_hist']]
    ax1.plot(gap_hist, color=bar_colors[idx], linewidth=2, label=f"{ds_name} (Opt={opt})")
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Optimum (0%)')
ax1.set_xlabel('Thế hệ')
ax1.set_ylabel('Gap so với Optimum (%)')
ax1.set_title('Đường hội tụ trung bình (Gap %)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
gap_costs_all = []
for ds_name in results_phase3.keys():
    opt = results_phase3[ds_name]['optimum']
    gap_costs = [(c - opt) / opt * 100 for c in results_phase3[ds_name]['costs']]
    gap_costs_all.append(gap_costs)

bp3 = ax2.boxplot(gap_costs_all, labels=list(results_phase3.keys()), patch_artist=True)
for patch, color in zip(bp3['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
ax2.set_ylabel('Gap so với Optimum (%)')
ax2.set_title('Phân phối kết quả sau 30 lần chạy (Gap %)')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase3_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: phase3_convergence.png")

# ---- Biểu đồ Gap theo quy mô ----
ds_names = list(results_phase3.keys())
n_sizes  = [results_phase3[d]['n']   for d in ds_names]
gaps     = [results_phase3[d]['gap'] for d in ds_names]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(ds_names, gaps, color=bar_colors[:len(ds_names)], edgecolor='black', width=0.5)
for bar, gap in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{gap:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Dataset (số thành phố tăng dần)')
ax.set_ylabel('Gap (%) so với Optimum')
ax.set_title(f'Sự suy giảm hiệu năng theo quy mô — [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]')
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('phase3_scalability_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: phase3_scalability_gap.png")


# ============================================================
# TỔNG KẾT: Kiểm chứng các Giả thuyết
# ============================================================

print("\n" + "=" * 65)
print("KIỂM CHỨNG GIẢ THUYẾT")
print("=" * 65)

# H1: OX tốt hơn PMX và CX (trung bình Best Cost thấp hơn)
avg_best = {cx: np.mean([min(results_phase1[(cx, m)]) for m in MUTATION_TYPES])
            for cx in CROSSOVER_TYPES}
h1_ok = avg_best['ox'] <= avg_best['pmx'] and avg_best['ox'] <= avg_best['cx']
print(f"\nH1 — OX tốt nhất về Best Cost: {'✅ ĐÚNG' if h1_ok else '❌ SAI'}")
for cx in CROSSOVER_TYPES:
    print(f"   {cx.upper()} avg best: {avg_best[cx]:.1f}")

# H2: Inversion tốt hơn Swap và Scramble
avg_mut = {m: np.mean([min(results_phase1[(cx, m)]) for cx in CROSSOVER_TYPES])
           for m in MUTATION_TYPES}
h2_ok = avg_mut['inversion'] <= avg_mut['swap'] and avg_mut['inversion'] <= avg_mut['scramble']
print(f"\nH2 — Inversion tốt nhất về Best Cost: {'✅ ĐÚNG' if h2_ok else '❌ SAI'}")
for m in MUTATION_TYPES:
    print(f"   {m:10s} avg best: {avg_mut[m]:.1f}")

# H3: [OX + Inversion] có Std Dev thấp nhất
all_stds = {k: np.std(v) for k, v in results_phase1.items()}
min_key  = min(all_stds, key=all_stds.get)
h3_ok    = min_key == ('ox', 'inversion')
std_ox_inv = all_stds.get(('ox', 'inversion'), float('inf'))
print(f"\nH3 — [OX+Inversion] ổn định nhất (Std thấp nhất): {'✅ ĐÚNG' if h3_ok else '❌ SAI'}")
print(f"   Std [OX+Inversion]    : {std_ox_inv:.2f}")
print(f"   Cấu hình Std thấp nhất: {min_key} = {all_stds[min_key]:.2f}")

print("\n" + "=" * 65)
print("KẾT LUẬN CHUNG")
print("=" * 65)
print(f"Cấu hình tốt nhất từ thực nghiệm: [{BEST_CROSSOVER.upper()} + {BEST_MUTATION}]")
print("\nScalability (Gap so với Optimum TSPLIB):")
for ds_name, info in results_phase3.items():
    print(f"  {ds_name:10s} ({info['n']:3d} TP): "
          f"Best={info['best']:.0f} | Gap={info['gap']:+.2f}% | Optimum={info['optimum']}")
print("=" * 65)
