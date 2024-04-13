# multilayer perceptron (MLP) using a Sequential model 
# Artificial Neural Network (ANN) using TensorFlow and Keras for binary classification on the heart disease dataset
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
    Dense(256, activation='relu', input_shape=(X.shape[1],)), #first hidden layer
    Dense(128, activation='relu'),  #second hidden layer
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

# Define a function to preprocess input data
def preprocess_input_data(data):
    # Convert data to numpy array
    data_array = np.array(data)
    # Reshape data array to match model input shape
    data_array = data_array.reshape(1, -1)
    # Normalize input features
    data_array = scaler.transform(data_array)
    return data_array
# Test data for a single person
test_data = [54, 1, 0, 120, 188, 0, 1, 113, 0, 1.4, 1, 1, 3]
# Preprocess the test data
processed_test_data = preprocess_input_data(test_data)
# Predict whether the person has heart disease or not
prediction = model.predict(processed_test_data)
# Print the prediction
if prediction[0] == 1 :
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")

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
# plt.text(80, 100, 'Team:\n\n1-Rewan Ahmed Ali\n2-sara abdelkader\n3-asmaa mohamed\n4-maryam jamal\n5-aya sabry\n6-alaa atef',
#          fontsize=12, color='#000E8C', style='oblique', bbox=dict(facecolor='#6A698C', alpha=0.5))
plt.show()


#Data Loading and Preprocessing
#    - The heart disease dataset (`heart.csv`) is loaded using pandas.
#    - Features are normalized using `StandardScaler`.
#    - The input features (`X`) and labels (`Y`) are prepared.

# Model Definition
#    - A Sequential model is created with three Dense layers.
#    - The input layer has 256 neurons with ReLU activation.
#    - The second hidden layer has 128 neurons with ReLU activation.
#    - The output layer has 1 neuron with Sigmoid activation for binary classification.

# Model Compilation and Training
#    - The model is compiled with the Adam optimizer and binary cross-entropy loss function.
#    - It is trained on the training data (`X_train`, `Y_train`) for 50 epochs with a batch size of 32.
#    - Validation data is specified using the `validation_split` parameter.

# Model Evaluation
#    - The model is evaluated on the test data (`X_test`, `Y_test`), and test loss and accuracy are printed.
#    - Classification report and confusion matrix are generated to evaluate model performance.

# Prediction
#    - A function `preprocess_input_data` is defined to preprocess input data for prediction.
#    - Test data for a single person is provided.
#    - The test data is preprocessed and passed through the model to predict whether the person has heart disease or not.
#    - The prediction is printed.

# Visualization
#    - Training and validation accuracy and loss are plotted over epochs using Matplotlib.
