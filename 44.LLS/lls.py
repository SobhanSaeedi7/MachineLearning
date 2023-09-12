import numpy as np

class LLS:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X_train.T, self.X_train)), self.X_train.T), self.Y_train)
        self.w = w
        return self.w

    def predict(self, X):
        return self.w * X