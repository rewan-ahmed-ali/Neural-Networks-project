import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define HebbianLearning class
class HebbianLearning:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(n_samples):
                y_predicted = np.dot(X[i], self.weights) + self.bias
                update = self.learning_rate * y_predicted * X[i] * y[i]
                self.weights += update
                self.bias += self.learning_rate * y_predicted * y[i]

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return np.where(y_predicted >= 0, 1, 0)

# Load the heart diseases dataset
heart_data = pd.read_csv("heart.csv")

# Splitting the features and target
X = heart_data.drop(columns='target').values
y = heart_data['target'].values  # Keep target values as they are

# Splitting the data into training and testing sets
def split_data(X, y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.delete(np.arange(n_samples), test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = split_data(X, y)

# Create and train the HebbianLearning model
hebbian_learning = HebbianLearning()
hebbian_learning.train(X_train, y_train)

# Accuracy on training data
y_train_pred = hebbian_learning.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)
print("Accuracy on training data:", train_accuracy)

# Accuracy on test data
y_test_pred = hebbian_learning.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print("Accuracy on test data:", test_accuracy)

# Building a predictive system
input_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]])

# Predict using the trained model
prediction = hebbian_learning.predict(input_data)

# Output the prediction
if prediction[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")
