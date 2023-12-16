import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, input_length, learning_rate, type_of_data, function="sigmoid"):
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.function = function
        self.type = type_of_data
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []


    def activation(self, x):
        if self.function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.function == "relu":
            return np.maximum(0, x)
        elif self.function == "tanh":
            return np.tanh(x)
        elif self.function == "linear":
            return x
        elif self.function == 'softmax':
            return np.exp(x) / np.sum(np.exp(x))
        else:
            raise Exception("Unknown Activation Function")

    def fit(self, X_train, X_test, Y_train, Y_test, epochs):
        for epoch in tqdm(range(epochs)):
            for x_train, y_train in zip(X_train, Y_train):
                y_pred = self.activation(x_train * self.weights + self.bias)
                loss_weights = (y_pred - y_train) * x_train
                loss_bias = (y_pred - y_train)
                self.weights = self.weights - self.learning_rate * loss_weights
                self.bias = self.bias - self.learning_rate * loss_bias

            loss_train, accuracy_train = self.evaluate(X_train, Y_train)
            loss_test, accuracy_test = self.evaluate(X_test, Y_test)

            self.train_losses.append(loss_train)
            self.train_accuracies.append(accuracy_train)
            self.test_losses.append(loss_test)
            self.test_accuracies.append(accuracy_test)


    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = self.activation(x_test * self.weights + self.bias)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def calculate_loss(self, X_test, Y_test, metric='mse'):
        y_pred = self.predict(X_test)
        if metric == 'mse':
            loss = np.mean((y_pred - Y_test) ** 2)
        elif metric == 'mae':
            loss = np.mean(np.abs(y_pred - Y_test))
        elif metric == 'rmse':
            loss = np.sqrt(np.mean((Y_test - Y_pred)**2))
        else:
            raise Exception('Unknown metric!')

        return loss
    
    def calculate_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        if self.type == 'classification':
            Y_pred = np.where(Y_pred > 0.5, 1, 0)
            accuracy = np.sum(Y_pred == Y_test)/ len(Y_test)
        elif self.type == 'regression':
            # R_squared metric
            RSS = np.sum((Y_test - Y_pred)**2)
            TSS = np.sum((Y_test - np.mean(Y_test))**2)
            accuracy = 1 - RSS/TSS
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test)
        accuracy = self.calculate_accuracy(X_test, Y_test)
        return loss, accuracy
