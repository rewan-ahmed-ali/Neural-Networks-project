import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class McCullochPittsNN:
    def __init__(self, input_size, inhibitory_constant):
        self.input_size = input_size
        self.inhibitory_constant = inhibitory_constant
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def activate(self, x):
        #n⋅w−p
        weighted_sum = np.dot(x, self.weights) - self.inhibitory_constant
        theta = 0  
        return 1 if weighted_sum >= theta else 0

    def forward(self, X):
        outputs = []
        for x in X:
            outputs.append(self.activate(x))
        return np.array(outputs)
inhibitory_constant = np.random.rand() 

Heart_Data = pd.read_csv("heart.csv")
X = Heart_Data.drop(columns='target', axis=1).values.astype('float32')
Y = Heart_Data['target'].values.astype('float32')

# train the model
model = McCullochPittsNN(input_size=X.shape[1], inhibitory_constant=inhibitory_constant)

# Make predictions
predictions = model.forward(X)

# Model Evaluation
accuracy = np.mean(Y == predictions)
print("Accuracy:", accuracy)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(range(len(Y)), Y, color='blue', label='Actual')
plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target')
plt.legend()
plt.show()
