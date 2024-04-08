import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the heart diseases (csv) dataset to a pandas DataFrame
Heart_Data = pd.read_csv("heart.csv")

# Split the features and target
X = Heart_Data.drop(columns='target', axis=1).values
Y = Heart_Data['target'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape the data for CNN input (assuming data is 1D)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test Accuracy:", test_accuracy)

# Predictions
predictions = model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
accuracy = accuracy_score(Y_test, predictions_binary)
print("Accuracy:", accuracy)

# Predict whether a person has heart disease or not
input_data = np.array([[43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]])
input_data = scaler.transform(input_data)
input_data = input_data.reshape(1, input_data.shape[1], 1)
prediction = model.predict(input_data)[0][0]

if prediction == 1:
    print("The person has heart disease.")
else:
    print("The person does not have heart disease.")
