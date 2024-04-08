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
