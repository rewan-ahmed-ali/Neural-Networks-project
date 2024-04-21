import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)  # Compute weighted sum for all inputs
        activations = np.where(weighted_sum >= self.threshold, 1, 0)  # Apply threshold
        return activations

def plot_decision_boundary(neuron, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = neuron.activate(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.xlabel('Age')
    plt.ylabel('Resting Blood Pressure')
    plt.title('Decision Boundary')
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.show()

# Read the dataset
data = pd.read_csv('heart.csv')

# Extract features and target
X = data[['age', 'trestbps']].values
y = data['target'].values

# Define weights and threshold for the neuron
weights = [0.5, 0.5]  # Adjust these weights as needed
threshold = 1  # Adjust threshold as needed

# Create a McCulloch-Pitts neuron
neuron = McCullochPittsNeuron(weights, threshold)

# Plot the decision boundary
plot_decision_boundary(neuron, X, y)

# Calculate accuracy
predictions = neuron.activate(X)
accuracy = np.mean(predictions == y) * 100
print("Accuracy: {:.2f}%".format(accuracy))

# Make a prediction for a new data point
new_data_point = np.array([[60, 130]])  # Adjust as needed
prediction = neuron.activate(new_data_point)
if prediction[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")
