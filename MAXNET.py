import numpy as np
import pandas as pd
from random import uniform

def activation_fn(x):
    if x > 0:
        return x
    return 0

# Read input data from heart.csv
data = pd.read_csv("heart.csv")

# Select a row (e.g., the first row)
selected_row = data.iloc[0, :-1]  # Exclude the last column (target)

# Convert the selected row to decimal numbers
activations = selected_row.astype(float).values

# Step 1 - Initialize weights and activations
m = len(activations)
k = 0.2

a_old = activations.copy()
a_new = []
count = 0

while True:  # Step 2
    temp = sum(a_old)
    
    for i in range(0, m):  # Step 3 - Update activations of each node
        value = a_old[i] - k * temp + k * a_old[i]
        a_new.append(activation_fn(value))
    
    a_old = a_new.copy()  # Step 4 - Save activations for use in each iteration
    
    # Print current activations for each epoch
    print('EPOCH {} - activations = {}'.format(count + 1, a_old))
    
    if sum(a_new) == max(a_new):  # Step 5 - Test for stopping condition
        break
    
    a_new = []
    count += 1

print('The final activations are {}'.format(a_new))
