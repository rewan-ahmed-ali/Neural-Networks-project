import numpy as np
import pandas as pd

data = pd.read_csv("heart.csv")

row_index = 0  

# Extract the selected row as a numpy array
X = np.array(data.iloc[row_index, :-1])

# Parameters for Mexican Hat Net
R1 = 2
R2 = 5
C1 = 0.4
C2 = -0.3

# Padding X with zeros
X_new = np.pad(X, (R2, R2), mode='constant')

# Creating the weight vector
w1 = np.array([C2] * (R2 - R1))
w2 = np.array([C1] * (2 * R1 + 1))
w3 = np.array([C2] * (R2 - R1))
w = np.concatenate((w1, w2, w3))

# Array Status as matrix
print("Array Status")
for i in range(len(X)):
    sum = 0
    k = -R1
    array_status = []
    while k <= R1:
        sum = sum + C1 * X_new[i + k]
        array_status.append(C1 * X_new[i + k])
        k = k + 1
    k = -R2
    while k < -R1:
        sum = sum + C2 * X_new[i + k]
        array_status.insert(0, C2 * X_new[i + k])
        k = k + 1
    k = R1 + 1
    while k <= R2:
        sum = sum + C2 * X_new[i + k]
        array_status.append(C2 * X_new[i + k])
        k = k + 1
    print(np.array(array_status))

# Applying Mexican Hat Net
output = np.zeros(len(X))

for i in range(0, len(X)):
    output[i] = np.dot(w, X_new[i:i + len(w)])
print('')
print("Output:", output)