import numpy as np
from Model import Model

# Lineer Regresyon Modeli
class LinearRegressionModel(Model):
    def __init__(self, x_train, y_train, x_test, y_test, learning_rate=0.01, epoch_limit=100, time_limit=None):
        super().__init__(x_train, y_train, x_test, y_test, learning_rate, epoch_limit, time_limit)
        self.weights = None

    def train(self):
        num_samples, num_features = self.x_train.shape
        self.weights = np.zeros(num_features + 1)
        self.x_train = np.c_[np.ones((num_samples, 1)), self.x_train]

        for epoch in range(self.epoch_limit):
            y_predicted = np.dot(self.x_train, self.weights)
            error = y_predicted - self.y_train

            # Gradients
            dw = (1 / num_samples) * np.dot(self.x_train.T, error)

            # Update weights and bias
            self.weights -= self.learning_rate * dw

    def predict(self, x):
        num_samples = x.shape[0]
        return np.dot(np.c_[np.ones((num_samples, 1)), x], self.weights)

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
        mse = np.mean((y_test - y_predicted) ** 2)
        mae = np.mean(np.abs(y_test - y_predicted))
        r_squared = 1 - (np.sum((y_test - y_predicted) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        return {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R Squared': r_squared
        }
