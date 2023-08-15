import numpy as np

class KNN:
    def __init__(self, K):
        self.k = K

    #Training
    def fit(self, X, Y):
        self.X_train= X
        self.Y_train= Y

    def euclidean_distance(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2)**2))

    def predict(self,New_X):
        Y = []
        for x in New_X:
            distances = []
            for x_train in self.X_train:
                d = self.euclidean_distance(x, x_train)
                distances.append(d)

            nearest_neighbors = np.argsort(distances)[0:self.k]
            bincount = np.bincount(self.Y_train[nearest_neighbors])
            y = np.argmax(bincount)
            Y.append(y)
        return Y

    def evaluate(self,X_test, Y_test):
        Y_pred = self.predict(X_test)
        accuracy = np.sum(Y_pred == Y_test)/(len(Y_test))
        return accuracy
