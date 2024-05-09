import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Perceptron

class Perceptron:
    
    def init(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    # heaviside activation function
    def activation(self, z):
        return np.heaviside(z, 0)  # Applying activation function element-wise

    def fit(self, X, y):
        n_features = X.shape[1]
        
        # Initializing weights and bias
        self.weights = np.zeros((n_features))
        self.bias = 0
        
        # Iterating until the number of epochs
        for epoch in range(self.epochs):
            
            # Traversing through the entire training set
            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias  # Dot product for each sample
                y_pred = self.activation(z)  # Passing through an activation function
                
                # Updating weights and bias
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred)
                
        return self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

# Load iris dataset
iris = load_iris() 
X = iris.data[:, (0, 1)]  # petal length, petal width
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Custom Perceptron
perceptron = Perceptron(0.001, 100)
perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)
print("Custom Perceptron Accuracy:", accuracy_score(pred, y_test))
print("Custom Perceptron Classification Report:")
print(classification_report(pred, y_test, digits=2))

# Scikit-learn Perceptron
sk_perceptron = Perceptron(learning_rate=0.1, epochs=100)

sk_perceptron.fit(X_train, y_train)
sk_perceptron_pred = sk_perceptron.predict(X_test)
print("Scikit-learn Perceptron Accuracy:", accuracy_score(sk_perceptron_pred, y_test))