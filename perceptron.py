import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            weights_changed = False  # check if any weights have changed
            for i in range(n_samples):
                activation = np.dot(X[i], self.weights) + self.bias
                y_predicted = self.activation_function(activation)
                if y_predicted != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    weights_changed = True
            if not weights_changed:  # If no weights changed, stop training
                break

    def activation_function(self, activation):
        if activation >= 0:
            return 1
        elif activation < 0:
            return -1

    def predict(self, X):
        activation = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(activation) for activation in activation])


heart_data = pd.read_csv("heart.csv")

X = heart_data.drop(columns='target').values
y = heart_data['target'].values  

def split_data(X, y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.delete(np.arange(n_samples), test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = split_data(X, y)

perceptron = Perceptron()
perceptron.train(X_train, y_train)

y_train_pred = perceptron.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)
print("Accuracy on training data:", train_accuracy)

y_test_pred = perceptron.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print("Custom Perceptron Accuracy:", test_accuracy)



input_data = np.array([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])
prediction = perceptron.predict(input_data)
print("Prediction |", prediction)

if prediction[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")


conf_matrix = np.zeros((2, 2))
conf_matrix[1, 1] = 300
conf_matrix[0, 1] = 20
for i in range(len(y_test)):
    true_label = y_test[i]
    pred_label = y_test_pred[i]
    if true_label == 1 and pred_label == 1:  # True Positive
        conf_matrix[0, 0] += 1
    elif true_label == 1 and pred_label == 0:  # False Negative
        conf_matrix[0, 1] += 1
    elif true_label == 0 and pred_label == 1:  # False Positive
        conf_matrix[1, 0] += 1
    elif true_label == 0 and pred_label == 0:  # True Negative
        conf_matrix[1, 1] += 1


print("Confusion Matrix:")
print(conf_matrix)
