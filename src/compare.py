import numpy as np
from src.neural_network import fitness_function
import random

def grid_search(param_ranges):
    results = []
    for lr in np.linspace(param_ranges["learning_rate"][0], param_ranges["learning_rate"][1], 3):
        for bs in param_ranges["batch_size"]:
            for ep in param_ranges["epochs"]:
                params = [lr, bs, ep]
                fitness = fitness_function(params)
                results.append((params, fitness))
    best = max(results, key=lambda x: x[1])
    return best[0], best[1]

def random_search(param_ranges, trials):
    results = []
    for _ in range(trials):
        params = [
            random.uniform(param_ranges["learning_rate"][0], param_ranges["learning_rate"][1]),
            random.choice(param_ranges["batch_size"]),
            random.choice(param_ranges["epochs"])
        ]
        fitness = fitness_function(params)
        results.append((params, fitness))
    best = max(results, key=lambda x: x[1])
    return best[0], best[1]

if __name__ == "__main__":
    param_ranges = {
        "learning_rate": (0.0001, 0.1),
        "batch_size": [32, 64, 128, 256],
        "epochs": [5, 10, 15]
    }
    print("Exploring Grid Search:")
    best_grid, grid_fitness = grid_search(param_ranges)
    print(f"Best Params: {best_grid}, Accuracy: {grid_fitness:.4f}")

    print("\nExploring Random Search:")
    best_random, random_fitness = random_search(param_ranges, trials=4)
    print(f"Best Params: {best_random}, Accuracy: {random_fitness:.4f}")