import numpy as np
from Model import Model
from sklearn.preprocessing import OneHotEncoder


# Sınıflandırma Modeli
class ClassificationModel(Model):
    def __init__(self, x_train, y_train, x_test, y_test, learning_rate=0.01, epoch_limit=100, time_limit=None):
        super().__init__(x_train, y_train, x_test, y_test, learning_rate, epoch_limit, time_limit)
        self.weights = None
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.y_train = self.one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        self.y_test = self.one_hot_encoder.transform(y_test.reshape(-1, 1))

    def train(self):
        num_samples, num_features = self.x_train.shape
        self.weights = np.zeros((len(self.one_hot_encoder.categories_[0]), num_features + 1))
        x_train_with_bias = np.c_[np.ones((num_samples, 1)), self.x_train]

        for epoch in range(self.epoch_limit):
            linear_output = np.dot(x_train_with_bias, self.weights.T)
            y_predicted = self._softmax(linear_output)
            error = y_predicted - self.y_train

            # Gradients
            dw = (1 / num_samples) * np.dot(error.T, x_train_with_bias)

            # Update weights
            self.weights -= self.learning_rate * dw

    def predict(self, x):
        num_samples = x.shape[0]
        x = np.c_[np.ones((num_samples, 1)), x]
        linear_output = np.dot(x, self.weights.T)
        y_predicted = self._softmax(linear_output)
        y_predicted = np.argmax(y_predicted, axis=1)
        return y_predicted

    def test(self):
        y_predicted = self.predict(self.x_test)
        return y_predicted

    def summarize(self):
        print(f"Weights (including bias): {self.weights}")

    def evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_predicted = self.predict(x_test)
        accuracy = np.mean(y_predicted == np.argmax(y_test, axis=1))

        return {
            'Accuracy': accuracy
        }

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
