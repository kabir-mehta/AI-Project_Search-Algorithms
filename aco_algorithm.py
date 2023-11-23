import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Objective function to maximize or minimize
def objective_function_3d(x, y, z):
    return np.sin(x) * np.cos(y) + np.exp(-(x**2 + y**2 + z**2) / 10)

def ant_colony_optimization_3d(max_iterations, num_ants, alpha, beta, rho):
    # Initialization
    best_solution = None
    best_fitness = float('-inf')

    # Randomly initialize ant positions in 3D space
    ants = np.random.rand(num_ants, 3)

    for iteration in range(max_iterations):
        # Evaluate fitness of each ant's position
        fitness_values = np.array([objective_function_3d(x, y, z) for x, y, z in ants])

        # Update best solution if a better one is found
        if np.max(fitness_values) > best_fitness:
            best_fitness = np.max(fitness_values)
            best_solution = ants[np.argmax(fitness_values)]

        # Update pheromone levels
        pheromones = np.ones(num_ants) / num_ants
        pheromones *= (1 - rho)

        for i in range(num_ants):
            pheromones[i] += rho * fitness_values[i] / np.sum(fitness_values)

        # Update ant positions using pheromone information
        for i in range(num_ants):
            probabilities = pheromones ** alpha * (1.0 / fitness_values) ** beta
            probabilities /= np.sum(probabilities)

            selected_ant = np.random.choice(num_ants, p=probabilities)
            ants[i] = ants[selected_ant]

    # Find the index of the ant with the best fitness after all iterations
    best_ant_index = np.argmax([objective_function_3d(x, y, z) for x, y, z in ants])
    best_solution = ants[best_ant_index]
    best_fitness = objective_function_3d(best_solution[0], best_solution[1], best_solution[2])

    # Plot the 3D surface of the objective function
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function_3d(X, Y, best_solution[2])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(best_solution[0], best_solution[1], best_solution[2], color='red', marker='o', s=100, label='Global Maximum')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    return best_solution, best_fitness

# Set the parameters
max_iterations = 1000
num_ants = 100
alpha = 2.0  # Pheromone influence
beta = 2.0   # Fitness influence
rho = 0.1    # Pheromone evaporation rate

# Run ACO algorithm
best_solution, best_fitness = ant_colony_optimization_3d(max_iterations, num_ants, alpha, beta, rho)

# Print results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)