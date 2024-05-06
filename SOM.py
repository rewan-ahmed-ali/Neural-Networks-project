from minisom import MiniSom
import numpy as np
import pandas as pd

heart_data = pd.read_csv("heart.csv")
data = heart_data.drop(columns='target').values
data_normalized = (data - data.mean()) / data.std()
grid_size = 3, 3  
input_dimensions = data_normalized.shape[1]
som = MiniSom(grid_size[0], grid_size[1], input_dimensions, sigma=0.3, learning_rate=0.5)
som.train_random(data_normalized, 100)
cluster_centroids = som.get_weights()
print("Cluster Centroids:")
print(cluster_centroids)
winning_neurons = np.array([som.winner(x) for x in data_normalized])
print("\nWinning Neurons (Cluster Assignments):")
print(winning_neurons)

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
