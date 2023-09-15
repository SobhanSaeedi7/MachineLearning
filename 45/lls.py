import numpy as np
from numpy.linalg import inv

class LLS:
    def __init__(self):
        pass

    def fit(self, X, Y):
        w = inv(X.T @ X)@ X.T @ Y
        self.w = w
        return self.w

    def predict(self, X):
        return self.w * X

    def evaluate(self, X_test, Y_test, metric='mae'):
        Y_pred = self.predict(X_test)

        if metric== 'mae':
            loss = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)

        elif metric == 'mse':
            loss = np.sum((Y_test - Y_pred)**2) / len(Y_test)

        elif metric == 'rmse':
            loss = np.sqrt(np.sum((Y_test - Y_pred)**2) / len(Y_test))

        return loss
        