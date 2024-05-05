import pandas as pd
import numpy as np

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)  # Compute weighted sum for all inputs
        activations = np.where(weighted_sum >= self.threshold, 1, 0)  # Apply threshold
        return activations

def confusion_matrix(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 50, 70
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 1 and pred == 0:
            FN += 1
    return TP, FP, TN, FN

data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1).values
y = data['target'].values

num_features = X.shape[1]
weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 

threshold = num_features * np.sum(weights > 0) - np.sum(weights)

neuron = McCullochPittsNeuron(weights, threshold)

predictions = neuron.activate(X)


TP, FP, TN, FN = confusion_matrix(y, predictions)
accuracy = (TP + TN) / (TP + TN + FP + FN)

new_data_point = np.array([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])  
prediction = neuron.activate(new_data_point)

# print("New data point:", new_data_point)

print("Accuracy:", accuracy)
print("Threshold:", threshold)
print("Confusion Matrix:")
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

print("Prediction |", prediction)
if prediction[0] == 1:
    print("The person is predicted to be positive for heart disease.")
else:
    print("The person is predicted not to have heart disease.")