import streamlit as st
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import sys
import os
def run_code(code):
    # Redirect stdout to capture print statements
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    # Execute the code
    try:
        exec(code)
    except Exception as e:
        return str(e)

    # Capture output and reset stdout
    output = sys.stdout.getvalue()
    sys.stdout = original_stdout

    return output

def plot_code(code):
    # Create a new figure for the plot
    fig, ax = plt.subplots()

    # Execute the code that generates the plot
    try:
        exec(code, globals(), locals())
    except Exception as e:
        st.warning(f"Error during code execution: {e}")
        return

    # Display the captured plot
    st.pyplot(fig)

# Prebuilt code snippets
code_snippets = [
    {
        'title': 'Example 1: ABC Algorithm',
        'code': """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Objective function to maximize or minimize
def objective_function_3d(x, y, z):
    return np.sin(x) * np.cos(y) + np.exp(-(x**2 + y**2 + z**2) / 10)

# Artificial Bee Colony (ABC) Algorithm for 3D functions
def abc_algorithm_3d(objective_function, n_agents, n_iterations, lower_bound, upper_bound, minimize=True):
    # Initialize the positions of the agents in 3D space
    agents = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_agents, 3))

    for iteration in range(n_iterations):
        # Evaluate the objective function for each agent
        fitness_values = np.array([objective_function(*agent) for agent in agents])

        # Find the index of the agent with the best fitness
        best_agent_index = np.argmin(fitness_values) if minimize else np.argmax(fitness_values)

        # Employed Bee Phase: Update the positions of employed bees
        employed_bees = agents.copy()
        for i in range(n_agents):
            phi = np.random.uniform(low=-1, high=1, size=3)
            j = np.random.choice(np.delete(range(n_agents), i))
            employed_bees[i] = agents[i] + phi * (agents[i] - agents[j])

            # Clip positions to stay within the bounds
            employed_bees[i] = np.clip(employed_bees[i], lower_bound, upper_bound)

            # Evaluate the objective function for the new position
            current_fitness = objective_function(*agents[i])
            new_fitness = objective_function(*employed_bees[i])

            # Update the position if the new fitness is better
            if (minimize and new_fitness < current_fitness) or (not minimize and new_fitness > current_fitness):
                agents[i] = employed_bees[i]

        # Onlooker Bee Phase: Update the positions based on the probability of selection
        fitness_values -= np.min(fitness_values)  # Shift fitness values to be non-negative
        probabilities = 1 / (1 + fitness_values)
        probabilities /= np.sum(probabilities)

        onlooker_bees = agents.copy()
        for i in range(n_agents):
            selected_index = np.random.choice(range(n_agents), p=probabilities)
            phi = np.random.uniform(low=-1, high=1, size=3)
            j = np.random.choice(np.delete(range(n_agents), selected_index))
            onlooker_bees[i] = agents[i] + phi * (agents[i] - agents[j])

            # Clip positions to stay within the bounds
            onlooker_bees[i] = np.clip(onlooker_bees[i], lower_bound, upper_bound)

            # Evaluate the objective function for the new position
            current_fitness = objective_function(*agents[i])
            new_fitness = objective_function(*onlooker_bees[i])

            # Update the position if the new fitness is better
            if (minimize and new_fitness < current_fitness) or (not minimize and new_fitness > current_fitness):
                agents[i] = onlooker_bees[i]

        # Scout Bee Phase: Randomly replace the positions of scout bees
        for i in range(n_agents):
            if np.random.rand() < 0.01:  # Probability of scout bee phase
                agents[i] = np.random.uniform(low=lower_bound, high=upper_bound, size=3)

    # Find the index of the agent with the best fitness after all iterations
    best_agent_index = np.argmin([objective_function(*agent) for agent in agents]) if minimize \
        else np.argmax([objective_function(*agent) for agent in agents])
    best_solution = agents[best_agent_index]
    best_fitness = objective_function(*best_solution)

    return best_solution, best_fitness

# Set the parameters
n_agents = 50
n_iterations = 100
lower_bound = -5
upper_bound = 5

# Maximize the 3D function
max_solution, max_fitness = abc_algorithm_3d(objective_function_3d, n_agents, n_iterations, lower_bound, upper_bound, minimize=False)
print(f"Global Maximum solution: {max_solution}, Global Maximum fitness: {max_fitness}")

# Minimize the 3D function
min_solution, min_fitness = abc_algorithm_3d(objective_function_3d, n_agents, n_iterations, lower_bound, upper_bound, minimize=True)
print(f"Global Minimum solution: {min_solution}, Global Minimum fitness: {min_fitness}")

# Visualization of the objective function
x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function_3d(X, Y, max_solution[2])  # Z is set to a constant value for 2D visualization

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(max_solution[0], max_solution[1], max_fitness, color='red', marker='o', s=100, label='Global Maximum')
ax.scatter(min_solution[0], min_solution[1], min_fitness, color='blue', marker='o', s=100, label='Global Minimum')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objective Function Value')
ax.legend()
plt.show()

        """,
    },
    {
        'title': 'Example 2: GWO Algorithm',
        'code': """
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

        """,
    },
    # Add more code snippets as needed
]

# Streamlit app
st.title("Interactive Code Runner")

# Code input boxes and Run buttons
for i, snippet in enumerate(code_snippets):
    st.header(snippet['title'])
    code_key = f"code_{i}"
    edited_code = st.text_area(f"Python code:", snippet['code'], key=code_key, height=200)

    if st.button(f"Run Now ({i + 1})"):
        # Run the code and capture the output
        output = run_code(edited_code)

        # Display output
        st.subheader("Output:")
        st.text(output)

        # Plot the graph
        plot_code(edited_code)
