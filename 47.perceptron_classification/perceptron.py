import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, learning_rate, input_length, function='sigmoid'):
        self.learning_rate = learning_rate
        self.function = function
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def activation(self, x):
        if self.function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.function == 'relu':
            return np.maximum(0, x)
        elif self.function == 'tanh':
            return np.tanh(x)
        elif self.function == 'linear':
            return x
        elif self.function == 'softmax':
            return np.exp(x) / np.sum(np.exp(x))
        else:
            raise Exception('Unknown activation function')

    def fit(self, X_train, X_test, Y_train, Y_test, epochs):
        for epoch in tqdm(range(epochs)):
            for x, y in zip(X_train, Y_train):
                #forward
                y_pred = self.activation(x * self.weights + self.bias)
                #back propagation
                error = y - y_pred
                #update
                self.weights = self.weights - self.learning_rate * error * x
                self.bias = self.bias - self.learning_rate * error 
            loss_train, accuracy_train = self.evaluate(X_train, Y_train)
            loss_test, accuracy_test = self.evaluate(X_test, Y_test)

            self.train_losses.append(loss_train)
            self.train_accuracies.append(accuracy_train) 

            self.test_losses.append(loss_test)
            self.test_accuracies.append(accuracy_test) 

            
    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = self.activation(x_test @ self.weights + self.bias)
            Y_pred.append(y_pred)
        return np.array(Y_pred)
    
    def calculate_loss(self, X_test, Y_test, metric='mae'):
        Y_pred = self.predict(X_test)
        if metric == 'mae':
            loss = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)
        elif metric == 'mse':
            loss = np.sum((Y_test - Y_pred)**2) / len(Y_test)
        elif metric == 'rmse':
            loss = np.sqrt(np.sum((Y_test - Y_pred)**2) / len(Y_test))
        else:
            raise Exception('Unknown metric!')

        return loss
    
    def calculate_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        accuracy = np.sum(Y_pred == Y_test)/ len(Y_test)
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test)
        accuracy = self.calculate_accuracy(X_test, Y_test)

        return loss, accuracy