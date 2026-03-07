import optuna
import numpy as np
import random
import argparse
import time

# =============================================================================
# 1. CORE GA CLASSES AND FUNCTIONS (Extracted from GA.ipynb)
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
        # 1.Evaluate fitness
        for individual in self.population:
            individual.fitness = self.fitness_func(individual.chromosome)

        # 2.Arrange population
        self.population.sort(key=lambda x : x.fitness, reverse=True)

        # 3. Selection and create new population - Elitism
        new_population = self.population[:self.elite_size]

        while len(new_population) < self.pop_size:
            # Select parent
            parent1 = self.selection()
            parent2 = self.selection()

            # Crossover
            child_chromosome = self.crossover_func(parent1.chromosome, parent2.chromosome)

            # Mutation
            child_chromosome = self.mutate_func(child_chromosome)

            new_population.append(Individual(child_chromosome))

        self.population = new_population
    
    # Tourament Selection
    def selection(self):
        sample = random.sample(self.population, self.tournament_k)
        return max(sample, key=lambda x : x.fitness)

def tsp_crossover_ox(parent1, parent2):
    size = len(parent1)
    # cut random array from parent1
    idx1, idx2 = sorted(random.sample(range(size), 2))
    child: list = [None] * size
    child[idx1:idx2+1] = parent1[idx1:idx2+1]

    # fill the rest of child with parent2
    genes_in_child = set(child[idx1:idx2+1])
    p2_pointer = (idx2+1)%size
    child_pointer = (idx2+1)%size
    
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

    p2_pos = {v:k for k, v in enumerate(parent2)}

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

CROSSOVER_FUNCS = {
    'ox': tsp_crossover_ox,
    'pmx': tsp_crossover_pmx,
    'cx': tsp_crossover_cx
}

def make_mutate_func(mutate_type, mutate_rate):
    def mutate(chromosome):
        if random.random() < mutate_rate:
            c = chromosome[:]
            if mutate_type == 'swap':
                i, j = random.sample(range(len(c)), 2)
                c[i], c[j] = c[j], c[i]
            elif mutate_type == 'inversion':
                i, j = sorted(random.sample(range(len(c)), 2))
                c[i:j+1] = c[i:j+1][::-1]
            elif mutate_type == 'scramble':
                i, j = sorted(random.sample(range(len(c)), 2))
                sub = c[i:j+1]
                random.shuffle(sub)
                c[i:j+1] = sub
            return c
        else:
            return chromosome
    return mutate

def make_mixed_mutate_func(mutation_probs, mutate_rate):
    """
    mutation_probs: dict, e.g. {'swap': 0.3, 'inversion': 0.5, 'scramble': 0.2}
    Sum of probs must equal 1.0
    Random a number in [0, 1), check which interval it falls into to choose mutation type.
    """
    p_swap = mutation_probs['swap']
    p_inversion = mutation_probs['inversion']
    # p_scramble = mutation_probs['scramble']  # implicit: the remaining range
    
    def mutate(chromosome):
        if random.random() < mutate_rate:
            r = random.random()
            c = chromosome[:]
            if r < p_swap:
                # Swap mutation
                i, j = random.sample(range(len(c)), 2)
                c[i], c[j] = c[j], c[i]
            elif r < p_swap + p_inversion:
                # Inversion mutation
                i, j = sorted(random.sample(range(len(c)), 2))
                c[i:j+1] = c[i:j+1][::-1]
            else:
                # Scramble mutation
                i, j = sorted(random.sample(range(len(c)), 2))
                sub = c[i:j+1]
                random.shuffle(sub)
                c[i:j+1] = sub
            return c
        return chromosome
    return mutate

def make_fitness_function(distance_matrix):
    def fitness(chromosome):
        total = 0.0
        n = len(chromosome)
        for i in range(n):
            total += distance_matrix[chromosome[i]][chromosome[(i+1)%n]]
        return 1/total
    return fitness

# =============================================================================
# 2. DATA LOADING FUNCTIONS
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
            dist_matrix[i][j] = int(np.sqrt(dx*dx + dy*dy)+0.5)
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
# 4. OPTUNA TUNING
# =============================================================================

def objective(trial, n_cities, dist_matrix):
    # Suggest hyperparameters
    pop_size = trial.suggest_int('pop_size', 50, 200, step=10)
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.4)
    elite_size = trial.suggest_int('elite_size', 1, 5)
    tournament_k = trial.suggest_int('tournament_k', 2, 10)
    crossover_type = trial.suggest_categorical('crossover_type', ['ox', 'pmx', 'cx'])
    
    # Suggest mutation probabilities that sum to 1.0
    p_swap = trial.suggest_float('p_swap', 0.0, 1.0)
    p_inversion = trial.suggest_float('p_inversion', 0.0, 1.0 - p_swap)
    p_scramble = 1.0 - p_swap - p_inversion
    mutation_probs = {
        'swap': p_swap,
        'inversion': p_inversion,
        'scramble': p_scramble,
    }
    
    n_runs = 3
    generations = 300
    
    costs = []
    for run in range(n_runs):
        cost = run_single_experiment(
            n_cities=n_cities,
            dist_matrix=dist_matrix,
            crossover_type=crossover_type,
            mutation_probs=mutation_probs,
            pop_size=pop_size,
            generation=generations,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_k=tournament_k,
            seed=run
        )
        costs.append(cost)
        
    avg_cost = np.mean(costs)
    return avg_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize GA parameters using Optuna")
    parser.add_argument("--dataset", type=str, default="data/kroA100.tsp", help="Path to TSPLIB dataset")
    parser.add_argument("--weight_type", type=str, default="EUC_2D", help="Edge weight type (EUC_2D or GEO)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    coords = load_tsplib(args.dataset)
    dist_matrix = build_dist_matrix(coords, args.weight_type)
    n_cities = len(coords)
    print(f"Number of cities: {n_cities}")
    
    study = optuna.create_study(direction="minimize", study_name="GA_Optimization")
    
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, n_cities, dist_matrix), n_trials=args.n_trials)
    end_time = time.time()
    
    print("\n" + "="*50)
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Average Best Cost): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Show mutation probabilities
    p_s = trial.params['p_swap']
    p_i = trial.params['p_inversion']
    p_sc = 1.0 - p_s - p_i
    print("\n  Mutation Probabilities (sum = 1.0):")
    print(f"    p_swap:      {p_s:.4f}")
    print(f"    p_inversion: {p_i:.4f}")
    print(f"    p_scramble:  {p_sc:.4f}")
    print("="*50)
