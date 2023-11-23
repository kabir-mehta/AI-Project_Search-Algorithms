import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function to maximize (change this to your desired function)
def Objective_function_3d(x, y, z):
    return (np.sin(x) * np.cos(y) + np.exp(-(x**2 + y**2 + z**2) / 10))

def firefly_algorithm_3d(n_fireflies, max_generations, alpha, beta, gamma, xmin, xmax, ymin, ymax, zmin, zmax):
    # Initialize fireflies randomly within the search space
    fireflies = np.random.rand(n_fireflies, 3) * np.array([xmax - xmin, ymax - ymin, zmax - zmin]) + np.array([xmin, ymin, zmin])
    intensities = np.zeros(n_fireflies)

    for generation in range(max_generations):
        # Evaluate the intensities of fireflies
        intensities = np.array([Objective_function_3d(x, y, z) for x, y, z in fireflies])

        # Update firefly positions
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if intensities[i] < intensities[j]:
                    r = np.sqrt((fireflies[i, 0] - fireflies[j, 0])**2 + (fireflies[i, 1] - fireflies[j, 1])**2 + (fireflies[i, 2] - fireflies[j, 2])**2)
                    beta_ij = beta * np.exp(-gamma * r**2)
                    fireflies[i, 0] += alpha * (fireflies[j, 0] - fireflies[i, 0]) + beta_ij * (np.random.rand() - 0.5)
                    fireflies[i, 1] += alpha * (fireflies[j, 1] - fireflies[i, 1]) + beta_ij * (np.random.rand() - 0.5)
                    fireflies[i, 2] += alpha * (fireflies[j, 2] - fireflies[i, 2]) + beta_ij * (np.random.rand() - 0.5)

        # Ensure fireflies stay within the search space
        fireflies[:, 0] = np.clip(fireflies[:, 0], xmin, xmax)
        fireflies[:, 1] = np.clip(fireflies[:, 1], ymin, ymax)
        fireflies[:, 2] = np.clip(fireflies[:, 2], zmin, zmax)

    # Find the best solution
    best_index = np.argmax(intensities)
    best_solution = fireflies[best_index]
    best_intensity = intensities[best_index]

    # Visualize the result
    x_values = np.linspace(xmin, xmax, 100)
    y_values = np.linspace(ymin, ymax, 100)
    z_values = np.linspace(zmin, zmax, 100)
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)
    W = Objective_function_3d(X, Y, Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fireflies[:, 0], fireflies[:, 1], intensities, color='blue', s=20, label='Fireflies')
    ax.scatter(best_solution[0], best_solution[1], best_solution[2], color='red', s=100, label='Best Solution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    return best_solution, best_intensity


# Parameters
n_fireflies = 30
max_generations = 150
alpha = 0.2  # Attraction coefficient
beta = 0.8   # Absorption coefficient
gamma = 0.5  # Randomness coefficient
xmin, xmax = -5, 5
ymin, ymax = -5, 5
zmin, zmax = -8, 8

# Run the Firefly Algorithm
best_solution, best_intensity = firefly_algorithm_3d(n_fireflies, max_generations, alpha, beta, gamma, xmin, xmax, ymin, ymax, zmin, zmax)


x = np.linspace(xmin, xmax, 100)
y = np.linspace(ymin, ymax, 100)
X, Y = np.meshgrid(x, y)
Z = Objective_function_3d(X, Y, best_solution[2])  # Z is set to a constant value for 2D visualization

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(best_solution[0], best_solution[1], best_solution[2], color='red', marker='o', s=100, label='Global Maximum')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objective Function Value')
ax.legend()
plt.show()

# Print the result
print("Best Solution:", best_solution)
print("Best Intensity:", best_intensity)