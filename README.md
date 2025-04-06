# GA Hyperparameter Optimization for MNIST

A personal project and also proposed it in the AiQuest'25 Datathon to explore genetic algorithms for optimizing neural network hyperparameters on MNIST, this project can be considred as the solution for the challenge. the challenge description that i made is in the LaTeX pdf.

## About
In this project, I used genetic algorithms (GAs) to tune the learning rate, batch size, and epochs of a PyTorch neural network for classifying MNIST digits. I also compared GA with grid and random search to understand their trade-offs.

## Features
- GA with tournament selection, crossover, and mutation.
- Simple PyTorch neural network for MNIST.
- Visualization of optimization progress.
- Comparison with grid and random search.

## Installation
1. Clone the repo:
   ```
   git clone https://github.com/ZakLr/Genetic-Algorithm-for-Neural-Network-Hyper-Parameters-Optimization
   ```

   ```
   cd ga-hyperparam-mnist
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```


## Usage
- Run GA optimization:
  ```
  python -m src.optimize
  ```

- Visualize results:
  ```
  python -m src.visualize
  ```

- Compare with other methods:
  ```
  python -m src.compare
  ```


## Results
The GA successfully found hyperparameters yielding high validation accuracy on MNIST. See ```
examples/best_solution.npz' for the best configuration.

## License
MIT License (see 'LICENSE' file).

## Contact
Zakaria Lourghi - [z.lourghi@esi-sba.dz](mailto:z.lourghi@esi-sba.dz)