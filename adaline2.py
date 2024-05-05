import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Adaline:
    def __init__(self, batch_size=None, learning_rate=0.001, n_iterations=50, random_state=1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.gen = np.random.RandomState(self.random_state)

    def _shuffle(self, X, y):
        reorder = self.gen.permutation(y.shape[0])
        return X[reorder], y[reorder]

    def predict(self, X, threshold=0.5):
        predictions = self.activation(self.net_input(X))
        predictions = np.where(predictions >= threshold, 1, 0)
        return predictions

    def activation(self, X):
        """Compute binary activation"""
        return np.where(X >= 0, 1, -1)

    def net_input(self, X):
        '''Calculates the net input'''
        return np.matmul(X, self.w_) + self.b_

    def _update_weights_bias(self, X, y):
        if self.batch_size is not None:
            idx_batches = [list(range(i, i + self.batch_size)) for i in range(0, X.shape[0], self.batch_size)]
        else:
            idx_batches = [list(range(0, X.shape[0]))]
        loss = 0.
        for idx_batch in idx_batches:
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            output = self.activation(self.net_input(X_batch))
            errors = y_batch - output
            self.w_ += self.learning_rate * 2. * np.matmul(X_batch.T, errors) / X_batch.shape[0]
            self.b_ += self.learning_rate * 2. * errors.mean()
            loss += np.dot(errors, errors)
        return loss

   
    def fit(self, X, y):
        self.w_ = self.gen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = float(0.)
        self.losses_ = []
        self.mse_ = []
        for epoch in range(self.n_iterations):
            X, y = self._shuffle(X, y)
            loss = self._update_weights_bias(X, y)
            mse = loss / len(y)
            print(f'epoch {epoch}: loss={loss:.4f}, MSE={mse:.4f}')
            self.losses_.append(loss)
            self.mse_.append(mse)

        # Print accuracy and final MSE
        final_mse = self.losses_[-1] / len(y)
        print("Final MSE:", final_mse)
        
    def predict_single(self, data):
        data = np.array(data)
        data_std = (data - X.mean()) / X.std()
        prediction = self.predict(data_std.values.reshape(1, -1))
        return prediction[0]

Heart_Disease_DataSet = pd.read_csv("heart.csv")
X = Heart_Disease_DataSet.drop(['target'], axis=1)
y = Heart_Disease_DataSet['target']
X_std = (X - X.mean()) / X.std()
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)


clf = Adaline(batch_size=None, learning_rate=0.02, n_iterations=250, random_state=1)
clf.fit(X_train.to_numpy(), y_train.to_numpy())


predictions = clf.predict(X_test.to_numpy())

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

def confusion_matrix_custom(y_true, y_pred):
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))   
    return np.array([[TN, FP], [FN, TP]])
print("Confusion Matrix:")
print(confusion_matrix_custom(y_test, predictions))
# Example of predicting whether a person has heart disease or not based on a single row of features
test_data = [43,0,0,132,341,1,0,136,1,3,1,0,3]
prediction = clf.predict_single(test_data)
if prediction == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted not to have heart disease.")

