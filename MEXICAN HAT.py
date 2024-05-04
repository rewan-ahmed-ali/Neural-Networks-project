import numpy as np
import pandas as pd

def activation_function(x):
    if x < 0:
        return 0
    elif 0 <= x <= 22:
        return x
    else:
        return 22

data = pd.read_csv("heart.csv")

row_index = 0  

# Extract the selected row as a numpy array
X = np.array(data.iloc[row_index, :-1])

# Parameters for Mexican Hat Net
R1 = 2  # Step 1
R2 = 5  # Step 1
C1 = 0.4  # Step 1
C2 = -0.3  # Step 1
x_max = 22  # Step 1

# Padding X with zeros
X_new = np.pad(X, (R2, R2), mode='constant')

# Creating the weight vector
w1 = np.array([C2] * (R2 - R1))  # Step 1
w2 = np.array([C1] * (2 * R1 + 1))  # Step 1
w3 = np.array([C2] * (R2 - R1))  # Step 1
w = np.concatenate((w1, w2, w3))  # Step 1

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

# Applying Mexican Hat Net with activation function
output = np.zeros(len(X))
t_max = 10  
t = 1  

while t < t_max:  # Step 3
    for i in range(0, len(X)):
        net_input = np.dot(w, X_new[i:i + len(w)])  # Step 4
        output[i] = activation_function(net_input)  # Step 5

        # Step 6: Save current activation functions in X_old
        X_old = np.copy(output)

    # Step 7: Increment iteration counter
    t += 1

    # Step 8: Test stopping condition
    if t >= t_max:
        break

print('')
print("Output:", output)



true_labels = np.array(data.iloc[row_index, -1])
predicted_labels = (output > 0).astype(int)
accuracy = np.mean(predicted_labels == true_labels)


print("Accuracy:", accuracy)

# Prediction
if predicted_labels[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")


import matplotlib.pyplot as plt
# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(X, color='blue')
plt.title('Input Data')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(output, color='green')
plt.title('Output of Mexican Hat Net')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()