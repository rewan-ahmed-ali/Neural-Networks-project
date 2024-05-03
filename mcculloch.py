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


data = pd.read_csv('heart.csv')

# Extract features and target
X = data.drop('target', axis=1).values
y = data['target'].values

num_features = X.shape[1]
weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 

# تحديد حاجز العتبة بناءً على عدد المدخلات والأوزان الموجبة والسالبة
threshold = num_features * np.sum(weights > 0) - np.sum(weights)

neuron = McCullochPittsNeuron(weights, threshold)

predictions = neuron.activate(X)
accuracy = np.mean(predictions == y)

new_data_point = np.array([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])  
prediction = neuron.activate(new_data_point)

# print("New data point:", new_data_point)
print("Prediction |", prediction)
print("Accuracy |", accuracy)
print("threshold |",threshold)
if prediction[0] == 1:
    print("The person is predicted to be positive for heart disease.")
else:
    print("The person is predicted not to have heart disease.")
