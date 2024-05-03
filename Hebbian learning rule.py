import numpy as np
import pandas as pd

class HebbianLearning:
    def __init__(self,  epochs=100):
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(n_samples):
                # y_predicted = np.dot(X[i], self.weights) + self.bias
                update =   X[i] * y[i]
                self.weights += update
                self.bias +=  y[i]

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return self.activation_function(y_predicted)
 
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


hebbian_learning = HebbianLearning()
hebbian_learning.train(X_train, y_train)

y_train_pred = hebbian_learning.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)
print("Accuracy on training data:", train_accuracy)

y_test_pred = hebbian_learning.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print("Accuracy on test data:", test_accuracy)


"""
هجرب علي بيانات صف من صفوف الداتا فمثلا الصف 12
target 1 """
input_data = np.array([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])
print("\nTesting the network:")
# print("\ninput data ",input_data)

actual_target = heart_data['target'].iloc[-6]
print("\tActual target for this input data:", actual_target)

prediction = hebbian_learning.predict(input_data)
print("\tPrediction |", prediction)
if prediction[0] == actual_target:
    print("\nThe prediction matches the actual target.")
    print("The person is predicted to have heart disease.")
else:
    print("\nThe prediction does not match the actual target. The accuracy is low.")
    print("The person is predicted not to have heart disease.")