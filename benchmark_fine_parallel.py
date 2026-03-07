"""
Benchmark: Fine-Grained Parallel GA vs Sequential GA.
This script parallelizes the **inner loop** of the GA (fitness evaluation and offspring generation)
using ThreadPoolExecutor, taking advantage of Python 3.14t (no-GIL).
Runs 30 times and averages to avoid statistical outliers.
"""
import numpy as np
import random
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# 1. CORE GA
# =============================================================================

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, elite_size, crossover_func, mutate_func, fitness_func, tournament_k=5, executor=None):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.fitness_func = fitness_func
        self.population: list = []
        self.tournament_k = tournament_k
        self.executor = executor  # ThreadPoolExecutor for fine-grained parallelism

    def create_initial_population(self, create_individual_func):
        for _ in range(self.pop_size):
            self.population.append(Individual(create_individual_func()))

    def _eval_fitness_chunk(self, chromosomes):
        return [self.fitness_func(c) for c in chromosomes]

    def _create_offspring_chunk(self, chunk_size):
        chunk = []
        for _ in range(chunk_size):
            parent1 = self.selection()
            parent2 = self.selection()
            child_chromosome = self.crossover_func(parent1.chromosome, parent2.chromosome)
            child_chromosome = self.mutate_func(child_chromosome)
            chunk.append(Individual(child_chromosome))
        return chunk

    def evolve(self):
        # 1. Evaluate fitness
        if self.executor is not None:
            chromosomes = [ind.chromosome for ind in self.population]
            n_workers = self.executor._max_workers
            chunk_size = len(chromosomes) // n_workers
            rem = len(chromosomes) % n_workers
            chunks = []
            start = 0
            for i in range(n_workers):
                end = start + chunk_size + (1 if i < rem else 0)
                chunks.append(chromosomes[start:end])
                start = end
            
            fitness_chunks = list(self.executor.map(self._eval_fitness_chunk, chunks))
            
            # Flatten
            fitnesses = [fit for chunk in fitness_chunks for fit in chunk]
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness = fit
        else:
            for individual in self.population:
                individual.fitness = self.fitness_func(individual.chromosome)

        # 2. Arrange population
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # 3. Selection and create new population - Elitism
        new_population = self.population[:self.elite_size]

        number_of_offspring = self.pop_size - self.elite_size
        
        # 4. Generate offspring
        if self.executor is not None:
            # Parallel offspring generation with chunking
            n_workers = self.executor._max_workers
            chunk_size = number_of_offspring // n_workers
            rem = number_of_offspring % n_workers
            chunks = [chunk_size + (1 if i < rem else 0) for i in range(n_workers)]
            
            offspring_chunks = list(self.executor.map(self._create_offspring_chunk, chunks))
            for chunk in offspring_chunks:
                new_population.extend(chunk)
        else:
            # Sequential offspring generation
            for _ in range(number_of_offspring):
                new_population.extend(self._create_offspring_chunk(1))

        self.population = new_population

    def selection(self):
        sample = random.sample(self.population, self.tournament_k)
        return max(sample, key=lambda x: x.fitness)

# --- Crossover & Mutation ---
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

CROSSOVER_FUNCS = {'ox': tsp_crossover_ox}

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

def build_dist_matrix(coords, edge_weight_type="EUC_2D"):
    return build_distance_matrix(coords) # GEO removed for brevity in this bench

# =============================================================================
# 3. EXPERIMENT RUNNER
# =============================================================================
def run_single_experiment(
    n_cities, dist_matrix,
    crossover_type='ox', mutation_probs=None,
    pop_size=100, generation=300, mutation_rate=0.05,
    elite_size=1, tournament_k=5, seed=None,
    executor=None):  # NEW: pass executor to GA

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
        elite_size=elite_size,
        executor=executor
    )

    ga.create_initial_population(lambda: random.sample(range(n_cities), n_cities))

    for _ in range(generation):
        ga.evolve()
        # Final fitness eval for elites/offspring combined
        if executor is not None:
            chromosomes = [ind.chromosome for ind in ga.population if ind.fitness == 0.0]
            if chromosomes:
                n_workers = executor._max_workers
                chunk_size = len(chromosomes) // n_workers
                rem = len(chromosomes) % n_workers
                chunks = []
                start = 0
                for i in range(n_workers):
                    end = start + chunk_size + (1 if i < rem else 0)
                    chunks.append(chromosomes[start:end])
                    start = end
                
                fitness_chunks = list(executor.map(lambda chk: [fitness_func(c) for c in chk], chunks))
                fitnesses = [fit for chunk in fitness_chunks for fit in chunk]
                
                idx = 0
                for ind in ga.population:
                    if ind.fitness == 0.0:
                        ind.fitness = fitnesses[idx]
                        idx += 1
        else:
            for ind in ga.population:
                if ind.fitness == 0.0:
                    ind.fitness = fitness_func(ind.chromosome)
                    
        ga.population.sort(key=lambda x: x.fitness, reverse=True)

    best_cost = 1.0 / ga.population[0].fitness
    return best_cost

# =============================================================================
# 4. BENCHMARK
# =============================================================================
OPTIMAL_PARAMS = {
    'pop_size': 190,
    'mutation_rate': 0.3663320151223965,
    'elite_size': 5,
    'tournament_k': 10,
    'crossover_type': 'ox',
    'mutation_probs': {'swap': 0.0723, 'inversion': 0.8859, 'scramble': 0.0418},
}

DATASETS = {
    'berlin52': {'path': 'data/berlin52.tsp', 'optimum': 7542,  'edge_weight_type': 'EUC_2D'},
    'kroA100':  {'path': 'data/kroA100.tsp',  'optimum': 21282, 'edge_weight_type': 'EUC_2D'},
}

N_RUNS = 30
GENERATIONS = 300
N_WORKERS = os.cpu_count() or 4

def run_benchmark(n_cities, dist_matrix, mode, executor=None):
    """Runs 30 experiments and records total time."""
    costs = []
    run_times = []
    start_total = time.time()
    for run in range(N_RUNS):
        t0 = time.time()
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
            executor=executor
        )
        run_times.append(time.time() - t0)
        costs.append(cost)
    total_time = time.time() - start_total
    
    avg_run_time = np.mean(run_times)
    best_cost = min(costs)
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    return best_cost, avg_cost, std_cost, total_time, avg_run_time

def main():
    print(f"Python: {sys.version}")
    try:
        gil_status = "Disabled (Free-Threaded)" if not sys._is_gil_enabled() else "Enabled"
    except AttributeError:
        gil_status = "Enabled (no free-threading support)"
    print(f"GIL: {gil_status} | CPU cores: {os.cpu_count()} | Thread workers: {N_WORKERS}")
    print(f"Runs per dataset: {N_RUNS} | Generations: {GENERATIONS}\n")

    header = f"{'Dataset':<10} | {'Mode':<15} {'Best':>8} {'Avg':>8} {'Gap(%)':>8} {'Avg Time/Run':>14} {'Total Time(s)':>14}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, info in DATASETS.items():
        coords = load_tsplib(info['path'])
        dist_matrix = build_dist_matrix(coords, info['edge_weight_type'])
        n_cities = len(coords)
        optimum = info['optimum']

        # 1. Sequential Mode (No executor)
        seq_best, seq_avg, seq_std, seq_tot, seq_avg_run = run_benchmark(n_cities, dist_matrix, "Sequential", executor=None)
        seq_gap = ((seq_best - optimum) / optimum) * 100.0

        # 2. Fine-grained Parallel Mode (Thread Pool)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            par_best, par_avg, par_std, par_tot, par_avg_run = run_benchmark(n_cities, dist_matrix, "Fine-Grained", executor=executor)
        
        par_gap = ((par_best - optimum) / optimum) * 100.0

        print(f"{name:<10} | {'Sequential':<15} {seq_best:>8.0f} {seq_avg:>8.1f} {seq_gap:>7.2f}% {seq_avg_run:>12.3f}s {seq_tot:>13.2f}s")
        print(f"{'':<10} | {'Fine-Grained':<15} {par_best:>8.0f} {par_avg:>8.1f} {par_gap:>7.2f}% {par_avg_run:>12.3f}s {par_tot:>13.2f}s")
        
        speedup_tot = seq_tot / par_tot if par_tot > 0 else 0
        speedup_run = seq_avg_run / par_avg_run if par_avg_run > 0 else 0
        print(f"{'':<10} | {'Speedup':<15} {'':>8} {'':>8} {'':>8} {speedup_run:>12.2f}x {speedup_tot:>13.2f}x")
        print(sep)

if __name__ == "__main__":
    main()
