import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, epochs=100):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            activations.append(self.sigmoid(np.dot(activations[-1], self.weights[i]) + self.biases[i]))
        return activations

    def backpropagation(self, X, y, activations):
        deltas = [None] * len(self.weights)
        deltas[-1] = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        for i in range(len(deltas)-2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y):
        for _ in range(self.epochs):
            activations = self.feedforward(X)
            self.backpropagation(X, y, activations)

    def predict(self, X, threshold=0.5):
        activations = self.feedforward(X)
        return np.where(activations[-1] >= threshold, 1, 0)

# Loading the heart diseases (csv) to a pandas DataFrame
Heart_Data = pd.read_csv("heart.csv")

# splitting the features and target
X = Heart_Data.drop(columns='target', axis=1).values
Y = Heart_Data['target'].values.reshape(-1, 1)

# splitting the data into the training data & test data
def split_data(X, Y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.delete(np.arange(n_samples), test_indices)
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Neural Network
neural_network = NeuralNetwork(layers=[X_train.shape[1], 5, 1], learning_rate=0.01, epochs=1000)
neural_network.train(X_train, Y_train)

# Testing different decision thresholds
thresholds = [0.3, 0.5, 0.7, 2]
for threshold in thresholds:
    predictions = neural_network.predict(X_test, threshold=threshold)
    accuracy = np.mean(predictions == Y_test)
    print(f"Accuracy on test data with threshold {threshold}: {accuracy}")

# Building a predictive system
input_data = np.array([[43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]])
prediction = neural_network.predict(input_data)
if prediction[0][0] == 1:
    print("The person has heart disease.")
else:
    print("The person does not have heart disease.")
