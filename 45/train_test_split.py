import numpy as np

def train_test_split(X, Y, test_size=0.5, shuffle=False):
    if len(X) != len(Y):
        raise ValueError('length of X_train and Y_train is not equal.')

    if shuffle:
        X, Y = shuffle_data(X, Y)

    if test_size < 1 :
        train_ratio = len(Y) - int(len(Y) * test_size)

        X_train, X_test = X[:train_ratio], X[train_ratio:] 
        Y_train, Y_test = Y[:train_ratio], Y[train_ratio:]

        return X_train, X_test, Y_train, Y_test

    elif test_size in range(1, len(Y)):
        X_train, X_test = X[test_size:], X[:test_size]
        Y_train, Y_test = Y[test_size:], Y[:test_size]

        return X_train, X_test, Y_train, Y_test

    else:
        raise ValueError('test-size value is out of range')


def shuffle_data(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    Data_num = np.arange(X.shape[0])
    np.random.shuffle(Data_num)

    return X[Data_num], Y[Data_num]


