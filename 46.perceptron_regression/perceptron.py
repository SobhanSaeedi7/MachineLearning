import numpy as np


class Perceptron:
    def __init__(self, w=np.random.rand(1, 1), b=np.random.rand(1, 1), lrw=0.0001, lrb=0.1, epochs=20):
        self.w = w
        self.b = b
        self.learning_rate_w = lrw
        self.learning_rate_b = lrb
        self.epochs = epochs
        self.losses = []

    def fit(self, X_train, Y_train):
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                y_pred = x * self.w + self.b
                error = y - y_pred

                loss = np.mean(np.abs(error))
                

                self.w = self.w + (error * x * self.learning_rate_w)
                self.b = self.b + (error * self.learning_rate_b)
            self.losses.append(loss)


    def predict(self, X_test):
        Y_pred = X_test * self.w + self.b

        return Y_pred

    def evaluate(self, X_test, Y_test, metric='mae'):

        Y_pred = X_test * self.w + self.b

        if metric == 'mae':
            loss = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)


        elif metric == 'mse':
            loss = np.sum((Y_test - Y_pred)**2) / len(Y_test)

        elif metric == 'rmse':
            loss = np.sqrt(np.sum((Y_test - Y_pred)**2) / len(Y_test))

        return loss


if  __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('Inputs/weight-height.csv')

    X = data[['Height']].values
    Y = data[['Weight']].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

    perceptron = Perceptron()

    perceptron.fit(X_train, Y_train)

    print(perceptron.predict([181]))

    print(perceptron.evaluate(X_test, Y_test))
    print(perceptron.evaluate(X_test, Y_test, metric='mse'))
    print(perceptron.evaluate(X_test, Y_test, metric='rmse'))




