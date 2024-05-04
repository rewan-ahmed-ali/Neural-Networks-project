import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class AdaptiveLinearNeuron(object):
    def __init__(self, rate=0.01, niter=10, tolerance=1e-5):
        self.rate = rate
        self.niter = niter
        self.tolerance = tolerance

    def fit(self, X, y):
        self.weight = np.random.random(1 + X.shape[1])  # Step 1
        self.bias = np.random.random()  # Step 1
        self.errors = []
        self.cost = []
        total_errors = []  

        largest_weight_change = np.inf
        iter_count = 0
        # While stopping condition is false, do steps 3 3– 7.
        while largest_weight_change > self.tolerance and iter_count < self.niter:  # Step 2
            cost_epoch = 0
            weight_change = 0

            for xi, target in zip(X, y):  # Step 3
                net_input = self.net_input(xi)  # Step 4
                output = self.activation(net_input)  # Step 4
                error = (target - output)  # Step 4
                self.weight[1:] += self.rate * xi.dot(error)  # Step 6
                self.bias += self.rate * error  # Step 6
                total_mean_square_error=(target-output)^2
                # Calculate the largest weight change
                weight_change = max(weight_change, np.max(np.abs(self.rate * xi.dot(error))))  # Step 7
                cost_epoch += 0.5 * error ** 2  # Step 7

            self.cost.append(cost_epoch)
            total_errors.append(cost_epoch)  # Store the total error for this epoch
            largest_weight_change = weight_change
            iter_count += 1

        return total_errors

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.bias  # Step 5

    def activation(self, X):
        """Compute binary activation"""
        return np.where(X >= 0, 1, -1)  # Step 5

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)  # Step 5

    def test(self, X):
        predictions = []
        for xi in X:  # Step 2
            net_input = self.net_input(xi)  # Step 3
            output = self.activation(net_input)  # Step 4
            predictions.append(output)  # Step 5
        return predictions


heart_data = pd.read_csv('heart.csv')
X = heart_data.drop(columns='target').values
y = heart_data['target'].values
y = np.where(y == 0, -1, 1)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Build and train the ADALINE model with normalized features
adaline = AdaptiveLinearNeuron(rate=0.01, niter=10, tolerance=1e-5)
errors = adaline.fit(X_train, y_train)
# Test the model
predictions = adaline.test(X_test)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

train_predictions = adaline.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

test_accuracy = accuracy_score(y_test, predictions)

print("Summary Results")
print("Epoch\tTotal Mean Square Error")
for epoch, error in enumerate(errors, start=1):
    print(f"Epoch {epoch}\t{error:10.2f}")

print(f"Error:\t {errors[-1]:.2f}")
