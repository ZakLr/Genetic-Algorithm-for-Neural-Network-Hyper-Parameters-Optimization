import numpy as np
from src.ga import generate_population, tournament_selection, crossover, mutate
from src.neural_network import fitness_function

def genetic_algorithm(population_size, generations, mutation_rate, param_ranges):
    population = generate_population(population_size, param_ranges)
    fitness_history = []

    print(f"Starting GA with population: {population}")
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")
        fitness_scores = [fitness_function(ind) for ind in population]
        fitness_history.append(max(fitness_scores))

        parents = tournament_selection(population, fitness_scores)
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1] if i + 1 < len(parents) else parents[i], parents[0]
            offspring1, offspring2 = crossover(parent1, parent2)
            next_population.append(mutate(offspring1, mutation_rate, param_ranges))
            next_population.append(mutate(offspring2, mutation_rate, param_ranges))

        population = next_population[:population_size]
        best_individual = max(zip(population, fitness_scores), key=lambda x: x[1])
        print(f"Best Hyperparameters: {best_individual[0]} | Accuracy: {best_individual[1]:.4f}")

    best_idx = np.argmax([fitness_function(ind) for ind in population])
    best_solution = population[best_idx]
    np.savez("examples/best_solution.npz", learning_rate=best_solution[0], 
             batch_size=best_solution[1], epochs=best_solution[2])
    np.save("examples/fitness_history.npy", fitness_history)
    return best_solution

if __name__ == "__main__":
    param_ranges = {
        "learning_rate": (0.0001, 0.1),
        "batch_size": [32, 64, 128, 256],
        "epochs": [5, 10, 15]  # Adjusted for a personal project scale
    }
    best_hyperparameters = genetic_algorithm(population_size=4, generations=3, mutation_rate=0.1, 
                                             param_ranges=param_ranges)
    print(f"\nFinal Best Hyperparameters: {best_hyperparameters}")