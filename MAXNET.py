import numpy as np
import pandas as pd
from random import uniform

class MaxNet:
    def __init__(self, data):
        self.data = data
        self.selected_row = data.iloc[0, :-1].astype(float).values
        self.m = len(self.selected_row)
        self.k = 0.2
        self.a_old = self.selected_row.copy()

    def activation_fn(self, x):
        if x > 0:
            return x
        return 0

    def train(self):
        a_new = []
        count = 0
        while True:
            temp = sum(self.a_old)
            for i in range(0, self.m):
                value = self.a_old[i] - self.k * temp + self.k * self.a_old[i]
                a_new.append(self.activation_fn(value))
            self.a_old = a_new.copy()
            print('EPOCH {} - activations = {}'.format(count + 1, self.a_old))
            if sum(a_new) == max(a_new):
                break
            a_new = []
            count += 1

    def predict_single(self, test_data):
        activation_sum = sum([test_data[i] * self.a_old[i] for i in range(self.m)])
        if activation_sum > 0:
            return 1
        return 0
    
    def calculate_accuracy(self, test_data, targets):
        correct = 0
        total = len(test_data)
        for i in range(total):
            prediction = self.predict_single(test_data[i])
            if prediction == targets[i]:
                correct += 1
        return correct / total

# Read input data from heart.csv
data = pd.read_csv("heart.csv")

# Extract features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize and train the MaxNet
model = MaxNet(data)
model.train()

# Test prediction on a single data point
test_data = [[43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]]
targets = [1]  # Assuming the target for the test_data is 1

prediction = model.predict_single(test_data[0])

# Calculate accuracy
accuracy = model.calculate_accuracy(test_data, targets)

# Output the prediction and accuracy
if prediction == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")

print("Accuracy: ",accuracy )
