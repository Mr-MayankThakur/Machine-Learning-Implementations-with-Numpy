import numpy as np

def feature_normalize(X, mean=None, sigma=None):
    """
    returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    the normalization is processed separately for each feature

    :param X: numpy array of shape (m,n)
        Training data
    :return: X_norm, mu, sigma
        X_norm matrix(m,n) - Normalised Feature matrix
        mu matrix(1,n) - mean of individual features
        sigma matrix(1,n) - standard deviation of individual features
    """
    if mean is None:
        mean = X.mean(0)

    if sigma is None:
        sigma = X.std(0)

    X_norm = (X-mean)/sigma

    return X_norm, mean, sigma


def add_bias_unit(X):
    """
    Adds bias unit to the training data.
    i.e.: adds column of 1's at the far left

    :param X: numpy array of shape (m,n)
        Training data
    :return: numpy array of shape (m, n+1)
        Training data with bias unit that is first column with 1's
    """

    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)