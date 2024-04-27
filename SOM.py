from minisom import MiniSom
import numpy as np
import pandas as pd

# Load the heart data from CSV
heart_data = pd.read_csv("heart.csv")

# Drop the target column since SOM is an unsupervised method
data = heart_data.drop(columns='target').values

# Normalize the data to improve training performance
data_normalized = (data - data.mean()) / data.std()

# Define the dimensions of the SOM grid
grid_size = 3, 3  # Grid size of 3x3 neurons

# Define the number of features (input dimensions)
input_dimensions = data_normalized.shape[1]

# Initialize the SOM
som = MiniSom(grid_size[0], grid_size[1], input_dimensions, sigma=0.3, learning_rate=0.5)

# Train the SOM on the normalized data
som.train_random(data_normalized, 100)  # Train for 100 iterations

# Get the cluster centroids (neuron weights)
cluster_centroids = som.get_weights()

# Print the cluster centroids
print("Cluster Centroids:")
print(cluster_centroids)

# Get the winning neuron (closest neuron) for each data point
winning_neurons = np.array([som.winner(x) for x in data_normalized])

# Print the winning neurons for each data point
print("\nWinning Neurons (Cluster Assignments):")
print(winning_neurons)

# Visualize the SOM
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot the distance map
plt.colorbar()

# Mark the cluster centroids
for i, centroid in enumerate(cluster_centroids.reshape(-1, input_dimensions)):
    plt.text(centroid[0] + 0.5, centroid[1] + 0.5, str(i + 1), color='red', ha='center', va='center')

# Mark the data points
for i, (x, y) in enumerate(winning_neurons):
    plt.text(x + 0.5, y + 0.5, str(i + 1), color='blue', ha='center', va='center')

plt.title('Self-Organizing Map Clustering')
plt.show()
