import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('Inputs/weight-height.csv')

X = data[['Height']].values
Y = data[['Weight']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.99)


fig, (ax1, ax2) = plt.subplots(1, 2)


w = np.random.rand(1, 1)
b = np.random.rand(1, 1)
learning_rate_w =  0.0001
learning_rate_b =  0.1
epochs = 20

losses = []

for epoch in range(epochs):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = Y_train[i]

        y_pred = x * w + b
        error = y - y_pred

        w = w + (error * x * learning_rate_w)
        b = b + (error * learning_rate_b)

        loss = np.mean(np.abs(error))
        losses.append(loss)

        Y_pred = X_train * w + b
        ax1.clear()
        ax1.scatter(X_train, Y_train, color='blue')
        ax1.plot(X_train, Y_pred, color='red')

        ax2.clear()
        ax2.plot(losses)
        plt.pause(0.01)



