# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:37:18 2023

@author: Kabir
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Objective function to maximize
def objective_function_3d(x, y, z):
    return np.sin(x) * np.cos(y) + np.exp(-(x**2 + y**2 + z**2) / 10)  # Negative of the sphere function for maximization

# Grey Wolf Optimizer (GWO) Algorithm for 3D functions
def gwo_algorithm_3d(objective_function, n_agents, n_iterations, lower_bound, upper_bound):
    # Initialize the positions of the wolves in 3D space
    wolves = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_agents, 3))

    for iteration in range(1, n_iterations + 1):
        a = 2 - iteration * (2 / n_iterations)  # Alpha parameter
        a2 = -1 + iteration * (-1 / n_iterations)  # Alpha2 parameter

        for i in range(n_agents):
            r1, r2 = np.random.rand(2)  # Random values between 0 and 1

            A1 = 2 * a * r1 - a  # Equation (3.3)
            C1 = 2 * r2  # Equation (3.4)

            D_alpha = abs(C1 * wolves[i] - wolves[np.argmax(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))])  # Equation (3.5)-part 1
            X1 = wolves[np.argmax(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))] - A1 * D_alpha  # Equation (3.6)-part 1

            r1, r2 = np.random.rand(2)

            A2 = 2 * a2 * r1 - a2  # Equation (3.3)
            C2 = 2 * r2  # Equation (3.4)

            D_beta = abs(C2 * wolves[i] - wolves[np.argsort(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))[1]])  # Equation (3.5)-part 2
            X2 = wolves[np.argsort(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))[1]] - A2 * D_beta  # Equation (3.6)-part 2

            r1, r2 = np.random.rand(2)

            A3 = 2 * a2 * r1 - a2  # Equation (3.3)
            C3 = 2 * r2  # Equation (3.4)

            D_delta = abs(C3 * wolves[i] - wolves[np.argsort(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))[2]])  # Equation (3.5)-part 3
            X3 = wolves[np.argsort(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))[2]] - A3 * D_delta  # Equation (3.5)-part 3

            wolves[i] = (X1 + X2 + X3) / 3  # Equation (3.7)

            # Clip positions to stay within the bounds
            wolves[i] = np.clip(wolves[i], lower_bound, upper_bound)



    # Find the index of the wolf with the best fitness after all iterations
    best_agent_index = np.argmax(objective_function(wolves[:, 0], wolves[:, 1], wolves[:, 2]))
    best_solution = wolves[best_agent_index]
    best_fitness = objective_function(best_solution[0], best_solution[1], best_solution[2])


    
    return best_solution, best_fitness

# Set the parameters
n_agents = 50
n_iterations = 100
lower_bound = -5
upper_bound = 5

# Maximize the 3D function
max_solution, max_fitness = gwo_algorithm_3d(objective_function_3d, n_agents, n_iterations, lower_bound, upper_bound)
print(f"Global Maximum solution: {max_solution}, Global Maximum fitness: {max_fitness}")

# Visualization of the objective function
x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function_3d(X, Y, max_solution[2]) # Negative of the sphere function for maximization

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(max_solution[0], max_solution[1], max_solution[2], color='red', marker='o', s=100, label='Global Maximum')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objective Function Value')
ax.legend()
plt.show()
