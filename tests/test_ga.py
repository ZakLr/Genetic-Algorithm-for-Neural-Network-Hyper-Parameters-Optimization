import pytest
from src.ga import generate_population, crossover

def test_generate_population():
    ranges = {"learning_rate": (0.0001, 0.1), "batch_size": [32, 64], "epochs": [5, 10]}
    pop = generate_population(2, ranges)
    assert len(pop) == 2
    assert len(pop[0]) == 3

def test_crossover():
    p1 = [0.01, 32, 5]
    p2 = [0.02, 64, 10]
    c1, c2 = crossover(p1, p2)
    assert len(c1) == 3
    assert c1[0] in [0.01, 0.02]