# Analysis of the Provided Code: Perceptron, not ANN
# The provided code defines and implements a Perceptron, which is a single-layer neural network and the simplest form of an artificial neuron. It's not a full Artificial Neural Network (ANN) as it lacks hidden layers that enable complex feature learning and non-linear decision boundaries.
# Here's a breakdown of the code:
# 1. Perceptron Class:
# __init__: Initializes the learning rate, epochs, weights (initially set to 0), and bias (initially set to 0).
# train: Trains the Perceptron on the provided data (X, y). It iterates through the data for a specified number of epochs, updating weights and bias based on the prediction error.
# predict: Makes predictions on new data by calculating the dot product of input features and weights, adding the bias, and applying a step function (thresholding) to classify the output as 0 or 1.
# 2. Data Loading and Preparation:
# Loads the "heart.csv" dataset using pandas.
# Separates features (X) and target variable (y).
# Splits the data into training and testing sets using the split_data function.
# 3. Training and Evaluation:
# Creates a Perceptron instance.
# Trains the Perceptron on the training data.
# Evaluates the accuracy on both training and testing data.
# 4. Prediction:
# Creates an example input data point.
# Uses the trained Perceptron to predict the output (presence or absence of heart disease).
# Key Points:
# The Perceptron is a linear binary classifier, meaning it can only learn linearly separable patterns.
# It updates its weights based on the misclassified examples, gradually moving the decision boundary to better separate the classes.
# Distinction from ANNs:
# ANNs typically have multiple layers with non-linear activation functions, enabling them to learn complex relationships and patterns in data.
# The Perceptron is limited to linear problems, while ANNs can tackle non-linear problems.

import numpy as np
import pandas as pd

class Perceptron:
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
                if y_predicted >= 0:
                    y_predicted = 1
                else:
                    y_predicted = 0

                update = self.learning_rate * (y[i] - y_predicted)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return np.where(y_predicted >= 0, 1, 0)

# Loading the heart diseases (csv) to a pandas DataFrame
Heart_Data = pd.read_csv("heart.csv")

# splitting the features and target
X = Heart_Data.drop(columns='target', axis=1).values
Y = Heart_Data['target'].values

# splitting the data into the training data & test data
def split_data(X, Y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.delete(np.arange(n_samples), test_indices)
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Perceptron
perceptron = Perceptron()
perceptron.train(X_train, Y_train)

# accuracy on training data
X_train_Prediction = perceptron.predict(X_train)
training_data_accuracy = np.mean(X_train_Prediction == Y_train)
print("Accuracy on training data: ", training_data_accuracy)

# accuracy on test data
X_test_Prediction = perceptron.predict(X_test)
test_data_accuracy = np.mean(X_test_Prediction == Y_test)
print("Accuracy on test data: ", test_data_accuracy)

# Building a predictive system.
input_data = np.array([[43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]])

prediction = perceptron.predict(input_data)
print(prediction)

if prediction[0] == 1:
    print("The person has heart disease")
else:
    print("The person does not have heart disease")
