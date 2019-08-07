import numpy as np

def feature_normalize(X):
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

    mu = X.mean(0)
    sigma = X.std(0)

    X_norm = (X-mu)/sigma

    return X_norm, mu, sigma