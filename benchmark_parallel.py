"""
Benchmark: Sequential vs Parallel GA using Python 3.14 Free-Threaded (No-GIL).
Compares both accuracy (Best, Avg, Std, Gap%) and execution time (speedup).
"""
import numpy as np
import random
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# 1. CORE GA (self-contained to avoid import issues across envs)
# =============================================================================

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, elite_size, crossover_func, mutate_func, fitness_func, tournament_k=5):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.fitness_func = fitness_func
        self.population: list = []
        self.tournament_k = tournament_k

    def create_initial_population(self, create_individual_func):
        for _ in range(self.pop_size):
            self.population.append(Individual(create_individual_func()))

    def evolve(self):
        for individual in self.population:
            individual.fitness = self.fitness_func(individual.chromosome)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = self.population[:self.elite_size]
        while len(new_population) < self.pop_size:
            parent1 = self.selection()
            parent2 = self.selection()
            child_chromosome = self.crossover_func(parent1.chromosome, parent2.chromosome)
            child_chromosome = self.mutate_func(child_chromosome)
            new_population.append(Individual(child_chromosome))
        self.population = new_population

    def selection(self):
        sample = random.sample(self.population, self.tournament_k)
        return max(sample, key=lambda x: x.fitness)

# --- Crossover operators ---
def tsp_crossover_ox(parent1, parent2):
    size = len(parent1)
    idx1, idx2 = sorted(random.sample(range(size), 2))
    child: list = [None] * size
    child[idx1:idx2+1] = parent1[idx1:idx2+1]
    genes_in_child = set(child[idx1:idx2+1])
    p2_pointer = (idx2+1) % size
    child_pointer = (idx2+1) % size
    item_added = 0
    while item_added < size - (idx2 - idx1 + 1):
        gene_from_p2 = parent2[p2_pointer]
        if gene_from_p2 not in genes_in_child:
            child[child_pointer] = gene_from_p2
            child_pointer = (child_pointer + 1) % size
            item_added += 1
        p2_pointer = (p2_pointer + 1) % size
    return child

def tsp_crossover_pmx(parent1, parent2):
    size = len(parent1)
    child: list = [None] * size
    idx1, idx2 = sorted(random.sample(range(size), 2))
    child[idx1:idx2+1] = parent1[idx1:idx2+1]
    p2_pos = {v: k for k, v in enumerate(parent2)}
    for i in range(idx1, idx2+1):
        gene_p2 = parent2[i]
        if gene_p2 not in child[idx1:idx2+1]:
            current_gene_p1 = parent1[i]
            target_idx = p2_pos[current_gene_p1]
            while idx1 <= target_idx <= idx2:
                new_gene_p1 = parent1[target_idx]
                target_idx = p2_pos[new_gene_p1]
            child[target_idx] = gene_p2
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child

def tsp_crossover_cx(parent1, parent2):
    size = len(parent1)
    child: list = [None] * size
    current_idx = 0
    while child[current_idx] is None:
        child[current_idx] = parent1[current_idx]
        val_in_p2 = parent2[current_idx]
        current_idx = parent1.index(val_in_p2)
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child

CROSSOVER_FUNCS = {'ox': tsp_crossover_ox, 'pmx': tsp_crossover_pmx, 'cx': tsp_crossover_cx}

# --- Mutation ---
def make_mixed_mutate_func(mutation_probs, mutate_rate):
    p_swap = mutation_probs['swap']
    p_inversion = mutation_probs['inversion']

    def mutate(chromosome):
        if random.random() < mutate_rate:
            r = random.random()
            c = chromosome[:]
            if r < p_swap:
                i, j = random.sample(range(len(c)), 2)
                c[i], c[j] = c[j], c[i]
            elif r < p_swap + p_inversion:
                i, j = sorted(random.sample(range(len(c)), 2))
                c[i:j+1] = c[i:j+1][::-1]
            else:
                i, j = sorted(random.sample(range(len(c)), 2))
                sub = c[i:j+1]
                random.shuffle(sub)
                c[i:j+1] = sub
            return c
        return chromosome
    return mutate

# --- Fitness ---
def make_fitness_function(distance_matrix):
    def fitness(chromosome):
        total = 0.0
        n = len(chromosome)
        for i in range(n):
            total += distance_matrix[chromosome[i]][chromosome[(i+1) % n]]
        return 1 / total
    return fitness

# =============================================================================
# 2. DATA LOADING
# =============================================================================
def load_tsplib(filepath):
    coords = []
    with open(filepath, 'r') as f:
        reading = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if line in ("EOF", ""):
                if reading:
                    break
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords

def build_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_matrix[i][j] = int(np.sqrt(dx*dx + dy*dy) + 0.5)
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix

def build_distance_matrix_geo(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    RRR = 6378.388
    PI = 3.141592
    lat_lon_rad = []
    for i in range(n):
        deg_lat = int(coords[i][0])
        min_lat = coords[i][0] - deg_lat
        rad_lat = PI * (deg_lat + 5.0 * min_lat / 3.0) / 180.0
        deg_lon = int(coords[i][1])
        min_lon = coords[i][1] - deg_lon
        rad_lon = PI * (deg_lon + 5.0 * min_lon / 3.0) / 180.0
        lat_lon_rad.append((rad_lat, rad_lon))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
                continue
            q1 = np.cos(lat_lon_rad[i][1] - lat_lon_rad[j][1])
            q2 = np.cos(lat_lon_rad[i][0] - lat_lon_rad[j][0])
            q3 = np.cos(lat_lon_rad[i][0] + lat_lon_rad[j][0])
            d = int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
            dist[i][j] = d
    return dist

def build_dist_matrix(coords, edge_weight_type="EUC_2D"):
    if edge_weight_type == "EUC_2D":
        return build_distance_matrix(coords)
    elif edge_weight_type == "GEO":
        return build_distance_matrix_geo(coords)

# =============================================================================
# 3. EXPERIMENT RUNNER
# =============================================================================
def run_single_experiment(
    n_cities, dist_matrix,
    crossover_type='ox',
    mutation_probs=None,
    pop_size=100,
    generation=300,
    mutation_rate=0.05,
    elite_size=1,
    tournament_k=5,
    seed=None):

    if mutation_probs is None:
        mutation_probs = {'swap': 0.0, 'inversion': 1.0, 'scramble': 0.0}

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fitness_func = make_fitness_function(dist_matrix)
    crossover_func = CROSSOVER_FUNCS[crossover_type]
    mutation_func = make_mixed_mutate_func(mutation_probs, mutation_rate)

    ga = GeneticAlgorithm(
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        mutate_func=mutation_func,
        crossover_func=crossover_func,
        fitness_func=fitness_func,
        tournament_k=tournament_k,
        elite_size=elite_size
    )

    ga.create_initial_population(lambda: random.sample(range(n_cities), n_cities))

    for _ in range(generation):
        ga.evolve()
        for individual in ga.population:
            if individual.fitness == 0.0:
                individual.fitness = fitness_func(individual.chromosome)
        ga.population.sort(key=lambda x: x.fitness, reverse=True)

    best_cost = 1.0 / ga.population[0].fitness
    return best_cost

# =============================================================================
# 4. BENCHMARK CONFIG
# =============================================================================
OPTIMAL_PARAMS = {
    'pop_size': 190,
    'mutation_rate': 0.3663320151223965,
    'elite_size': 5,
    'tournament_k': 10,
    'crossover_type': 'ox',
    'mutation_probs': {
        'swap': 0.07228022069012682,
        'inversion': 0.8858594404265957,
        'scramble': 1.0 - 0.07228022069012682 - 0.8858594404265957,
    },
}

DATASETS = {
    'burma14':  {'path': 'data/burma14.tsp',  'optimum': 3323,  'edge_weight_type': 'GEO'},
    'berlin52': {'path': 'data/berlin52.tsp', 'optimum': 7542,  'edge_weight_type': 'EUC_2D'},
    'kroA100':  {'path': 'data/kroA100.tsp',  'optimum': 21282, 'edge_weight_type': 'EUC_2D'},
}

N_RUNS = 30
GENERATIONS = 300
N_WORKERS = os.cpu_count() or 4

# =============================================================================
# 5. SEQUENTIAL vs PARALLEL RUNNERS
# =============================================================================
def run_benchmark_sequential(n_cities, dist_matrix):
    """Run N_RUNS experiments sequentially."""
    costs = []
    for run in range(N_RUNS):
        cost = run_single_experiment(
            n_cities=n_cities, dist_matrix=dist_matrix,
            crossover_type=OPTIMAL_PARAMS['crossover_type'],
            mutation_probs=OPTIMAL_PARAMS['mutation_probs'],
            pop_size=OPTIMAL_PARAMS['pop_size'],
            generation=GENERATIONS,
            mutation_rate=OPTIMAL_PARAMS['mutation_rate'],
            elite_size=OPTIMAL_PARAMS['elite_size'],
            tournament_k=OPTIMAL_PARAMS['tournament_k'],
            seed=run,
        )
        costs.append(cost)
    return costs

def run_benchmark_parallel(n_cities, dist_matrix, n_workers):
    """Run N_RUNS experiments in parallel using ThreadPoolExecutor."""
    def _run_one(run_seed):
        return run_single_experiment(
            n_cities=n_cities, dist_matrix=dist_matrix,
            crossover_type=OPTIMAL_PARAMS['crossover_type'],
            mutation_probs=OPTIMAL_PARAMS['mutation_probs'],
            pop_size=OPTIMAL_PARAMS['pop_size'],
            generation=GENERATIONS,
            mutation_rate=OPTIMAL_PARAMS['mutation_rate'],
            elite_size=OPTIMAL_PARAMS['elite_size'],
            tournament_k=OPTIMAL_PARAMS['tournament_k'],
            seed=run_seed,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_one, run) for run in range(N_RUNS)]
        costs = [f.result() for f in futures]
    return costs

# =============================================================================
# 6. MAIN
# =============================================================================
def compute_metrics(costs, optimum):
    best = min(costs)
    avg = np.mean(costs)
    std = np.std(costs)
    gap = ((best - optimum) / optimum) * 100.0
    return best, avg, std, gap

def main():
    # Print environment info
    print(f"Python: {sys.version}")
    try:
        gil_status = "Disabled (Free-Threaded)" if not sys._is_gil_enabled() else "Enabled"
    except AttributeError:
        gil_status = "Enabled (no free-threading support)"
    print(f"GIL: {gil_status}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Thread workers: {N_WORKERS}")
    print(f"Runs per dataset: {N_RUNS} | Generations: {GENERATIONS}")
    print()

    # Header
    header = f"{'Dataset':<10} {'N':>4} | {'Mode':<12} {'Best':>8} {'Avg':>8} {'Std':>8} {'Gap(%)':>8} {'Time(s)':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    summary_rows = []

    for name, info in DATASETS.items():
        try:
            coords = load_tsplib(info['path'])
            dist_matrix = build_dist_matrix(coords, info['edge_weight_type'])
            n_cities = len(coords)
            optimum = info['optimum']
        except FileNotFoundError:
            print(f"{name:<10} File not found: {info['path']}")
            continue

        # --- Sequential ---
        t0 = time.time()
        seq_costs = run_benchmark_sequential(n_cities, dist_matrix)
        seq_time = time.time() - t0
        seq_best, seq_avg, seq_std, seq_gap = compute_metrics(seq_costs, optimum)

        # --- Parallel ---
        t0 = time.time()
        par_costs = run_benchmark_parallel(n_cities, dist_matrix, N_WORKERS)
        par_time = time.time() - t0
        par_best, par_avg, par_std, par_gap = compute_metrics(par_costs, optimum)

        speedup = seq_time / par_time if par_time > 0 else float('inf')

        print(f"{name:<10} {n_cities:>4} | {'Sequential':<12} {seq_best:>8.0f} {seq_avg:>8.1f} {seq_std:>8.1f} {seq_gap:>7.2f}% {seq_time:>8.2f}")
        print(f"{'':>16}| {'Parallel':<12} {par_best:>8.0f} {par_avg:>8.1f} {par_std:>8.1f} {par_gap:>7.2f}% {par_time:>8.2f}")
        print(f"{'':>16}| {'Speedup':<12} {'':>8} {'':>8} {'':>8} {'':>8} {speedup:>7.2f}x")
        print(f"{'':>16}| {'Optimum':<12} {optimum:>8}")
        print(sep)

        summary_rows.append({
            'name': name, 'n': n_cities, 'optimum': optimum,
            'seq_best': seq_best, 'seq_time': seq_time,
            'par_best': par_best, 'par_time': par_time,
            'speedup': speedup,
        })

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Optimum':>8} | {'Seq Best':>9} {'Seq Time':>9} | {'Par Best':>9} {'Par Time':>9} | {'Speedup':>8}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['name']:<10} {r['optimum']:>8} | {r['seq_best']:>9.0f} {r['seq_time']:>8.2f}s | {r['par_best']:>9.0f} {r['par_time']:>8.2f}s | {r['speedup']:>7.2f}x")
    print("-" * 80)

    print(f"\nParameters: pop={OPTIMAL_PARAMS['pop_size']}, mr={OPTIMAL_PARAMS['mutation_rate']:.4f}, "
          f"elite={OPTIMAL_PARAMS['elite_size']}, k={OPTIMAL_PARAMS['tournament_k']}, "
          f"cx={OPTIMAL_PARAMS['crossover_type']}")
    mp = OPTIMAL_PARAMS['mutation_probs']
    print(f"Mutation probs: swap={mp['swap']:.4f}, inv={mp['inversion']:.4f}, scr={mp['scramble']:.4f}")

if __name__ == "__main__":
    main()
