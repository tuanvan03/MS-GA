"""
Benchmark script: Test Optuna-optimized GA parameters on multiple TSP datasets.
Computes Best Fitness, Gap (%), Std, and Time (s) for each dataset.
"""
import numpy as np
import random
import time

# Import GA components from optuna_tune
from optuna_tune import (
    load_tsplib,
    build_dist_matrix,
    run_single_experiment,
)

# =============================================================================
# 1. OPTIMAL PARAMETERS (from Optuna tuning)
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
        'scramble': 1.0 - 0.07228022069012682 - 0.8858594404265957,  # ~0.0419
    },
}

# =============================================================================
# 2. DATASETS
# =============================================================================
DATASETS = {
    'burma14':  {'path': 'data/burma14.tsp',  'optimum': 3323,  'edge_weight_type': 'GEO'},
    'berlin52': {'path': 'data/berlin52.tsp', 'optimum': 7542,  'edge_weight_type': 'EUC_2D'},
    'kroA100':  {'path': 'data/kroA100.tsp',  'optimum': 21282, 'edge_weight_type': 'EUC_2D'},
}

# =============================================================================
# 3. BENCHMARK
# =============================================================================
N_RUNS = 30
GENERATIONS = 300

def main():
    print(f"{'Dataset':<12} {'N':>4} {'Best':>10} {'Avg':>10} {'Std':>10} {'Gap(%)':>8} {'Time(s)':>8}")
    print("-" * 70)

    for name, info in DATASETS.items():
        try:
            coords = load_tsplib(info['path'])
            dist_matrix = build_dist_matrix(coords, info['edge_weight_type'])
            n_cities = len(coords)
            optimum = info['optimum']
        except FileNotFoundError:
            print(f"{name:<12} File not found: {info['path']}")
            continue

        costs = []
        start_time = time.time()

        for run in range(N_RUNS):
            cost = run_single_experiment(
                n_cities=n_cities,
                dist_matrix=dist_matrix,
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

        elapsed = time.time() - start_time

        best_cost = min(costs)
        avg_cost = np.mean(costs)
        std_cost = np.std(costs)
        gap = ((best_cost - optimum) / optimum) * 100.0

        print(f"{name:<12} {n_cities:>4} {best_cost:>10.1f} {avg_cost:>10.1f} {std_cost:>10.1f} {gap:>7.2f}% {elapsed:>8.2f}")

    print("-" * 70)
    print(f"\nParameters used:")
    for k, v in OPTIMAL_PARAMS.items():
        if k == 'mutation_probs':
            print(f"  {k}:")
            for mk, mv in v.items():
                print(f"    {mk}: {mv:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"  generations: {GENERATIONS}")
    print(f"  n_runs: {N_RUNS}")

if __name__ == "__main__":
    main()
