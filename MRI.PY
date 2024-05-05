import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Activation function (bipolar)
def activation(x):
    return np.where(x >= 0, 1, -1)

def confusion_matrix(actual, predicted):
    true_positives = np.sum(np.logical_and(actual == 1, predicted == 1))
    false_positives = np.sum(np.logical_and(actual == -1, predicted == 1))
    true_negatives = np.sum(np.logical_and(actual == -1, predicted == -1))
    false_negatives = np.sum(np.logical_and(actual == 1, predicted == -1))
    
    return np.array([[true_positives, false_positives],
                     [false_negatives, true_negatives]])

class MADALINE:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights_hidden = np.random.randn(self.input_dim, self.hidden_dim)
        self.biases_hidden = np.random.randn(self.hidden_dim)
        self.weights_output = np.random.randn(self.hidden_dim)
        self.bias_output = np.random.randn()
    
    def train(self, X_train, y_train, max_iterations=1000):
        for _ in range(max_iterations):
            weight_change = 0
            for i in range(len(X_train)):
                # Forward pass
                hidden_output = np.dot(X_train[i], self.weights_hidden) + self.biases_hidden
                hidden_activations = activation(hidden_output)
                output = np.dot(hidden_activations, self.weights_output) + self.bias_output
                # Compute error
                error = y_train[i] - output
                # Update weights 
                if error != 0:
                    weight_change += 1
                    if y_train[i] == 1:
                        closest_hidden_index = np.argmin(np.abs(hidden_output))
                        self.weights_hidden[:, closest_hidden_index] += self.learning_rate * (1 - hidden_activations[closest_hidden_index]) * X_train[i]
                        self.biases_hidden[closest_hidden_index] += self.learning_rate * (1 - hidden_activations[closest_hidden_index])
                    elif y_train[i] == -1:
                        positive_hidden_indices = np.where(hidden_output > 0)[0]
                        self.weights_hidden[:, positive_hidden_indices] += self.learning_rate * (-1 - hidden_activations[positive_hidden_indices]) * X_train[i][:, np.newaxis]
                        self.biases_hidden[positive_hidden_indices] += self.learning_rate * (1 - hidden_activations[positive_hidden_indices])
                    # Update output weights
                    self.weights_output += self.learning_rate * error * hidden_activations
                    self.bias_output += self.learning_rate * error
            if weight_change == 0:
                break
    
    def predict(self, X):
        hidden_output = np.dot(X, self.weights_hidden) + self.biases_hidden
        hidden_activations = activation(hidden_output)
        return activation(np.dot(hidden_activations, self.weights_output) + self.bias_output)
    


data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1).values
y = data['target'].replace(0, -1).values  # Convert 0 labels to -1 for bipolar representation

madaline = MADALINE(input_dim=X.shape[1], hidden_dim=2, output_dim=1, learning_rate=0.1)
madaline.train(X, y)

# Test MADALINE
predictions = madaline.predict(X)
conf_matrix = confusion_matrix(y, predictions)
accuracy = np.mean(predictions == y)
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y, predictions, target_names=["No heart disease", "heart disease"])
print("Classification Report:")
print(report)

def preprocess_input_data(data, mean, std):
    data = np.array(data)
    data_std = (data - mean) / std
    return data_std

test_data = [
    [58, 0, 0, 100, 248, 0, 0, 122, 0, 1, 1, 0, 2]
]
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
processed_test_data = preprocess_input_data(test_data, mean, std)
predictions = madaline.predict(processed_test_data)
for i, prediction in enumerate(predictions):
    if prediction >= 0.5:
        print("The person is predicted to have heart disease.")
    else:
        print(" The person is predicted not to have heart disease.")
