import numpy as np
import pandas as pd

# Mexican Hat Net
R1 = 2
R2 = 5
C1 = 0.4
C2 = -0.3
tmax = 10

# Load heart disease dataset from CSV
heart_data = pd.read_csv("D:\\last heart\\heart.csv")

# Preprocess the dataset
X = heart_data.iloc[:, :-1].values  # Features
y = heart_data.iloc[:, -1].values  # Target variable

# Padding zeros around the feature array
Xnew = np.pad(X, ((0, 0), (R2, R2)), mode='constant')

print("Input: ", Xnew)

# Creating a weight vector
w1 = np.full(R2 - R1, C2)
w2 = np.full(2 * R1 + 1, C1)
w3 = np.full(R2 - R1, C2)
w = np.concatenate((w1, w2, w3))

print("Weights: ", w)

# Empty list to store predicted outputs
predictions = []

# Sliding Window Technique
for i in range(len(Xnew) - len(w) + 1):  # Adjusted range to fit the length of the weight vector
    ans = np.dot(w, Xnew[i:i + len(w)])
    predictions.append(ans)

# Apply threshold or classification rule to determine positive/negative for heart disease
threshold = 0.5  # Adjust the threshold as per your needs
predicted_labels = []

for p in predictions:
    if np.any(p >= threshold):
        predicted_labels.append(1)  # Positive
    else:
        predicted_labels.append(0)  # Negative

if predicted_labels[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")

# Calculate accuracy
accuracy = sum(1 for true, pred in zip(y, predicted_labels) if true == pred) / len(y)
print("Accuracy:", accuracy)

# Calculate confusion matrix
confusion_matrix = np.zeros((2, 2))
for true, pred in zip(y, predicted_labels):
    confusion_matrix[true][pred] += 1

print("Confusion Matrix:")
print(confusion_matrix)


import matplotlib.pyplot as plt
# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(X, color='blue')
plt.title('Input Data')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(predictions, color='green')
plt.title('Output of Mexican Hat Net')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()