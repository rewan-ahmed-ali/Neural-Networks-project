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


data = pd.read_csv('heart.csv')

# Extract features and target
X = data[['age', 'trestbps']].values
y = data['target'].values

weights = [0.5, 0.5]  
threshold = 0  


neuron = McCullochPittsNeuron(weights, threshold)

predictions = neuron.activate(X)
accuracy = np.mean(predictions == y)
print("Accuracy: ",accuracy)


new_data_point = np.array([[60, 130]])  
prediction = neuron.activate(new_data_point)
if prediction[0] == 1:
    print("The person is predicted to be positive for heart disease.")
else:
    print("The person is predicted not to have heart disease.")

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

plot_decision_boundary(neuron, X, y)
