import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Define the 3D function to maximize
def objective_function(x):
    return np.sin(x[0]) * np.cos(x[1]) + np.exp(-(x[0]**2 + x[1]**2 + x[2]**2) / 10)
def sinusoid(x):
    return np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2])

def particle_swarm_optimization(func, bounds, num_particles=100, max_iter=100, inertia_weight=0.5, cognitive_coefficient=1.5, social_coefficient=1.5):
    dim = len(bounds)
    
    # Initialize particles randomly within the bounds
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_particles, dim))
    
    # Initialize velocities randomly
    velocities = np.random.rand(num_particles, dim)
    
    # Initialize personal best positions and values
    personal_best_positions = particles.copy()
    personal_best_values = np.array([func(p) for p in personal_best_positions])
    
    # Initialize global best position and value
    global_best_index = np.argmax(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]

    # Store the history of particles for visualization
    particles_history = [particles.copy()]

    # Main PSO loop
    for _ in range(max_iter):
        # Update particle velocities
        r1, r2 = np.random.rand(num_particles, dim), np.random.rand(num_particles, dim)
        velocities = (inertia_weight * velocities +
                      cognitive_coefficient * r1 * (personal_best_positions - particles) +
                      social_coefficient * r2 * (global_best_position - particles))
        
        # Update particle positions
        particles += velocities
        
        # Clip positions to stay within bounds
        particles = np.clip(particles, bounds[:, 0], bounds[:, 1])
        
        # Update personal best positions and values
        current_values = np.array([func(p) for p in particles])
        update_personal_best = current_values > personal_best_values
        personal_best_positions[update_personal_best] = particles[update_personal_best]
        personal_best_values[update_personal_best] = current_values[update_personal_best]
        
        # Update global best position and value
        global_best_index = np.argmax(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        # Store the current particles for visualization
        particles_history.append(particles.copy())
    
    return global_best_position, global_best_value, particles_history

# Define the bounds for each dimension
bounds = np.array([[-100, 100], [-100, 100], [-100, 100]])

# Run PSO and store the history of particles for visualization
result_position, result_value, particles_history = particle_swarm_optimization(objective_function, bounds)

# Define a function to visualize the optimization process
def plot_optimization_process(particles_history, func, global_best_position, bounds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D structure of the objective function
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y, np.zeros_like(X)])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolors='k', linewidth=0.5)

    # Plot the particles at each iteration
    for i, particles in enumerate(particles_history):
        ax.scatter(particles[:, 0], particles[:, 1], func(particles.T), label=f'Iteration {i + 1}')

    # Mark the global best position
    ax.scatter(global_best_position[0], global_best_position[1], func(global_best_position), color='red', s=100, label='Global Best')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    ax.set_title('PSO Optimization Process with Global Best')
    ax.legend()

    plt.show()

# Plot the optimization process
plot_optimization_process(particles_history, objective_function,result_position,bounds)

def plot_surface_with_global_best(func, global_best_position, bounds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D structure of the objective function
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y, np.zeros_like(X)])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolors='k', linewidth=0.5)

    # Mark the global best position
    ax.scatter(global_best_position[0], global_best_position[1], func(global_best_position), color='red', s=100, label='Global Best')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    ax.set_title('Objective Function with Global Best Position')
    ax.legend()

    plt.show()

# Plot the 3D surface of the objective function with the global best position
plot_surface_with_global_best(objective_function, result_position, bounds)

# Print the result
print("Optimal Position:", result_position)
print("Optimal Value:", result_value)  # Convert back to maximization