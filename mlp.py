# importing modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset file heart.csv
Heart_Data = pd.read_csv("heart.csv")

# Cast the records into float values
X = Heart_Data.drop(columns='target', axis=1).values.astype('float32')
Y = Heart_Data['target'].values.astype('float32')  # Ensure labels are float

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the network architecture using Keras 
#Sequential = feed forward network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
    # Dense(1, activation='linear') # Output layer with linear activation for binary classification
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test loss:", test_loss, "| Test accuracy:", test_accuracy)
# # Evaluate the network
# Y_pred = model.predict(X_test)
# y_pred = (Y_pred > 0.5).astype(np.float32)  # Convert probabilities to binary predictions
# print("Classification Report:")
# print(classification_report(Y_test, y_pred))

# Evaluate the network
print("[INFO] Evaluating network...")
Y_pred = model.predict(X_test)
y_pred = np.round(Y_pred).flatten()  # Round predictions to 0 or 1
print("Classification Report:")
print(classification_report(Y_test, y_pred))

# Create a Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

# Plot the training loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
# # Add text to the plot
# plt.text(80, 100, 'Team:\n\n1-Rewan Ahmed Ali\n2-sara abdelkader\n3-maryam jamal\n4-aya sabry\n5-alaa atef\n6-asmaa mohamed',
#          fontsize=12, color='#000E8C', style='oblique', bbox=dict(facecolor='#6A698C', alpha=0.5))

plt.show()

