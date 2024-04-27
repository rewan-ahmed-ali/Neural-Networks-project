import numpy as np
import pandas as pd

def activation_function(x):
    if x >= 0:
        return x
    else:
        return 0

def maxnet(input_data, epsilon=0.01, max_iterations=1000):
    m = len(input_data)  # Total number of nodes
    a_old = np.full(m, epsilon)  # Initialize activations with small values
    w = np.zeros((m, m))  # Initialize weights
    
    for i in range(m):
        for j in range(m):
            if i != j:
                w[i, j] = 1 / (m - 1)  # Adjust weights to ensure connectivity
    
    for iteration in range(max_iterations):
        a_new = np.zeros_like(a_old)  # Initialize new activations
        
        # Update activations of each node
        for j in range(m):
            sum_activation = 0
            for k in range(m):
                if k != j:
                    activation = activation_function(a_old[k] - epsilon) * w[k, j]
                    sum_activation += activation
            a_new[j] = sum_activation
        
        # Save activations for the next iteration
        a_old = np.copy(a_new)
        
        # Test stopping condition
        if np.count_nonzero(a_new) <= 1:
            break
        
        # Print activations for debugging
        print("Iteration", iteration+1, "Activations:", a_new)
    
    return a_new

# Read the input data from heart.csv
data = pd.read_csv("heart.csv")

# Select a row (e.g., the first row)
row_index = 0
selected_row = data.iloc[row_index, :-1]  # Exclude the last column (target)

# Convert the selected row to decimal numbers
input_data = selected_row.astype(float).values

# Apply the MAXNET algorithm
epsilon = 0.01
activations = maxnet(input_data, epsilon)

print("Final activations:", activations)
