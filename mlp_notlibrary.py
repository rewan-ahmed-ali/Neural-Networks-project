import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset file heart.csv
Heart_Data = pd.read_csv("heart.csv")

# Cast the records into float values
X = Heart_Data.drop(columns='target', axis=1).values.astype('float32')
Y = Heart_Data['target'].values.astype('float32')  # Ensure labels are float

# Normalize features manually
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Define the network architecture using Keras 
#Sequential = feed forward network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)), #first hidden layer
    Dense(128, activation='relu'),  #second hidden layer
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a function to split data
def train_test_split_custom(X, Y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_samples = int(X.shape[0] * test_size)
    X_test = X[indices[:test_samples]]
    Y_test = Y[indices[:test_samples]]
    X_train = X[indices[test_samples:]]
    Y_train = Y[indices[test_samples:]]
    return X_train, X_test, Y_train, Y_test

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split_custom(X, Y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test loss:", test_loss, "| Test accuracy:", test_accuracy)
# Model Evaluation
print("[INFO] Evaluating network...")
Y_pred = model.predict(X_test)
y_pred = np.round(Y_pred).flatten()  # Round predictions to 0 or 1

# Define a function to calculate classification report
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

classification_result = calculate_classification_report(Y_test, y_pred)

# Printing classification report in the desired format
print("Classification Report:")
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

# Create a Confusion Matrix
def confusion_matrix_custom(y_true, y_pred):
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])

print("Confusion Matrix:")
print(confusion_matrix_custom(Y_test, y_pred))

# Define a function to preprocess input data
def preprocess_input_data(data, mean, std):
    # Convert data to numpy array
    data_array = np.array(data)
    # Reshape data array to match model input shape
    data_array = data_array.reshape(1, -1)
    # Normalize input features manually
    data_array = (data_array - mean) / std
    return data_array

# Test data for a single person
test_data = [54, 1, 0, 120, 188, 0, 1, 113, 0, 1.4, 1, 1, 3]

# Preprocess the test data
processed_test_data = preprocess_input_data(test_data, mean, std)

# Predict whether the person has heart disease or not
prediction = model.predict(processed_test_data)

# Print the prediction
if prediction[0] == 1:
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
plt.show()
