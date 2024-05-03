import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
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

                if y_predicted != y[i]:
                    self.weights = self.weights + self.learning_rate * (y[i] * X[i])
                    self.bias = self.bias + self.learning_rate * (y[i])

    def activation_function(self, y_predicted):
        if y_predicted > 0:
            return 1
        elif y_predicted == 0:
            return 0
        else:
            return -1

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(y) for y in y_predicted])


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















def calculate_classification_report(y_true, y_pred):
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    precision_0 = TN / (TN + FP)
    recall_0 = TN / (TN + FN)
    f1_score_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    precision_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    f1_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return {
        '0.0': {'precision': precision_0, 'recall': recall_0, 'f1-score': f1_score_0, 'support': len(y_true) - np.sum(y_true)},
        '1.0': {'precision': precision_1, 'recall': recall_1, 'f1-score': f1_score_1, 'support': np.sum(y_true)},
        'accuracy': accuracy,
        'macro avg': {'precision': (precision_0 + precision_1) / 2, 'recall': (recall_0 + recall_1) / 2, 'f1-score': (f1_score_0 + f1_score_1) / 2, 'support': len(y_true)},
        'weighted avg': {'precision': (precision_0 * (len(y_true) - np.sum(y_true)) + precision_1 * np.sum(y_true)) / len(y_true), 
                         'recall': (recall_0 * (len(y_true) - np.sum(y_true)) + recall_1 * np.sum(y_true)) / len(y_true), 
                         'f1-score': (f1_score_0 * (len(y_true) - np.sum(y_true)) + f1_score_1 * np.sum(y_true)) / len(y_true), 
                         'support': len(y_true)}
    }

classification_result = calculate_classification_report(y_test, y_test_pred)
print("\nClassification Report:")
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("", "precision", "recall", "f1-score", "support"))
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("0.0", f"{classification_result['0.0']['precision']:.2f}", 
                                                  f"{classification_result['0.0']['recall']:.2f}", 
                                                  f"{classification_result['0.0']['f1-score']:.2f}", 
                                                  classification_result['0.0']['support']))
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("1.0", f"{classification_result['1.0']['precision']:.2f}", 
                                                  f"{classification_result['1.0']['recall']:.2f}", 
                                                  f"{classification_result['1.0']['f1-score']:.2f}", 
                                                  classification_result['1.0']['support']))
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("accuracy", "", "", f"{classification_result['accuracy']:.2f}", ""))
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("macro avg", f"{classification_result['macro avg']['precision']:.2f}", 
                                                  f"{classification_result['macro avg']['recall']:.2f}", 
                                                  f"{classification_result['macro avg']['f1-score']:.2f}", 
                                                  classification_result['macro avg']['support']))
print("{:<45} {:<12} {:<12} {:<12} {:<12}".format("weighted avg", f"{classification_result['weighted avg']['precision']:.2f}", 
                                                  f"{classification_result['weighted avg']['recall']:.2f}", 
                                                  f"{classification_result['weighted avg']['f1-score']:.2f}", 
                                                  classification_result['weighted avg']['support']))

# """هجرب علي بيانات صف من صفوف الداتا فمثلا الصف الاول 
# target بتاعه 0
# فالشخص تنبؤ بتاعه المفروض يطلع معندوش مرض قلب
# """
# # Building a predictive system
# input_data = np.array([[54, 1, 0, 120, 188, 0, 1, 113, 0, 1.4, 1, 1, 3]])

# # Predict using the trained Perceptron
# prediction = perceptron.predict(input_data)

# # Output the prediction
# if prediction[0] == 1:
#     print("The person is predicted to have heart disease.")
# else:
#     print("The person is predicted not to have heart disease.")
