import numpy as np
import matplotlib.pyplot as plt

def visualize_results(fitness_path="examples/fitness_history.npy"):
    fitness_history = np.load(fitness_path)

    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, label="Validation Accuracy")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("GA Hyperparameter Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_results()