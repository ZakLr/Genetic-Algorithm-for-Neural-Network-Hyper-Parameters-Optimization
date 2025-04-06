import random
import numpy as np

def generate_population(size, param_ranges):
    """Generate initial population with random hyperparameters."""
    return [
        [
            round(random.uniform(param_ranges["learning_rate"][0], param_ranges["learning_rate"][1]), 4),
            random.choice(param_ranges["batch_size"]),
            random.choice(param_ranges["epochs"])
        ]
        for _ in range(size)
    ]

def tournament_selection(population, fitness_scores, tournament_size=2):
    """Select parents using tournament selection."""
    tournament_size = min(tournament_size, len(population))
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parent1, parent2):
    """Single-point crossover for hyperparameters."""
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2

def mutate(offspring, mutation_rate, param_ranges):
    """Mutate hyperparameters with given probability."""
    if random.random() < mutation_rate:
        param_to_mutate = random.randint(0, len(offspring) - 1)
        if param_to_mutate == 0:  # learning_rate
            offspring[param_to_mutate] = round(random.uniform(param_ranges["learning_rate"][0], 
                                                             param_ranges["learning_rate"][1]), 4)
        elif param_to_mutate == 1:  # batch_size
            offspring[param_to_mutate] = random.choice(param_ranges["batch_size"])
        elif param_to_mutate == 2:  # epochs
            offspring[param_to_mutate] = random.choice(param_ranges["epochs"])
    return offspring