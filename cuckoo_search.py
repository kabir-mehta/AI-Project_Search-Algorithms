import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sin(x[0]) * np.cos(x[1]) + np.exp(-(x[0]**2 + x[1]**2 + x[2]**2) / 10)

def levy_flight(size, beta=1.5, scale=0.1):
    sigma = (np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = scale * np.random.randn(size)
    v = np.random.randn(size)
    step = u / np.abs(v) ** (1 / beta) * sigma
    return step

def cuckoo_search_maximization(func, bounds, num_cuckoos=100, max_iter=10000, pa=0.25, alpha=0.5, beta=1.5, scale=0.1):
    dim = len(bounds)
    
    # Initialize cuckoos randomly within the bounds
    cuckoos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_cuckoos, dim))
    
    # Evaluate initial positions
    values = np.array([func(c) for c in cuckoos])
    best_cuckoo_index = np.argmax(values)
    best_cuckoo = cuckoos[best_cuckoo_index].copy()
    best_value = values[best_cuckoo_index]
    
    # Main Cuckoo Search loop
    for iteration in range(max_iter):
        # Generate new solutions (cuckoos) via Levy flights
        step_sizes = levy_flight(num_cuckoos, beta, scale)
        new_cuckoos = cuckoos + alpha * step_sizes[:, np.newaxis] * np.random.randn(num_cuckoos, dim)
        
        # Clip new solutions to stay within bounds
        new_cuckoos = np.clip(new_cuckoos, bounds[:, 0], bounds[:, 1])
        
        # Evaluate new solutions
        new_values = np.array([func(c) for c in new_cuckoos])
        
        # Select the top pa proportion of cuckoos for replacement
        num_replace = int(pa * num_cuckoos)
        replace_indices = np.argsort(new_values)[-num_replace:]
        
        # Replace the worst cuckoos with new solutions
        cuckoos[replace_indices] = new_cuckoos[replace_indices]
        values[replace_indices] = new_values[replace_indices]
        
        # Update the best cuckoo
        current_best_index = np.argmax(values)
        if values[current_best_index] > best_value:
            best_cuckoo = cuckoos[current_best_index].copy()
            best_value = values[current_best_index]

        # Adjust alpha and scale dynamically
        alpha *= 0.99  # Decrease alpha over time
        scale *= 1.01  # Increase scale over time

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best Value: {best_value}")

    
    return best_cuckoo, best_value
def plot_objective_function_with_max_position(func, max_position, bounds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D structure of the objective function
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y, np.zeros_like(X)])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolors='k', linewidth=0.5)

    # Mark the maximum position found by Cuckoo Search
    ax.scatter(max_position[0], max_position[1], func(max_position), color='red', s=100, label='Max Position')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    ax.set_title('Objective Function with Max Position')
    ax.legend()

    plt.show()

# Define the bounds for each dimension
bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])

# Run Cuckoo Search for maximization
max_position, max_value = cuckoo_search_maximization(objective_function, bounds)
plot_objective_function_with_max_position(objective_function, max_position, bounds)

# Print the results
print("Maximum Position:", max_position)
print("Maximum Value:", max_value)