import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess data
Heart_Data = pd.read_csv("heart.csv")
X = Heart_Data.drop(columns='target', axis=1).values.astype('float32')
Y = Heart_Data['target'].values.astype('float32')  # Ensure labels are float
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Define the network architecture using Keras 
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)), #first hidden layer
    Dense(128, activation='relu'),  #second hidden layer
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

# Define the loss function
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
history = model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X, Y)
print("Test loss:", test_loss, "| Test accuracy:", test_accuracy)

# Model Evaluation
print("[INFO] Evaluating network...")
Y_pred = model.predict(X)
y_pred = np.round(Y_pred).flatten()

# Calculate classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(Y, y_pred))

def confusion_matrix_custom(y_true, y_pred):
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])

print("Confusion Matrix:")
print(confusion_matrix_custom(Y, y_pred))

# Preprocess input data
def preprocess_input_data(data, mean, std):
    data_array = np.array(data)
    data_array = data_array.reshape(1, -1)
    data_array = (data_array - mean) / std
    return data_array

# Predict new data
test_data = [54, 1, 0, 120, 188, 0, 1, 113, 0, 1.4, 1, 1, 3]
processed_test_data = preprocess_input_data(test_data, mean, std)
prediction = model.predict(processed_test_data)
if prediction[0] == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()
