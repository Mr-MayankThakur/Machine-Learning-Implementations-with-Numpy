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


def map_feature(X, degree, bias_unit=True):
    """
    Maps the input features to multiple new features using polynomial expansion.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    :param X: numpy array of size (m,n)
        Feature matrix with m training examples and n features

    :param degree: integer
        Degree is the highest number till which you want to expand the features

    :param bias_unit: bool
        If True a Bias unit column is added to the dataset.

    :return: numpy array (m,>n)
        Expanded Feature matrix with more features

    Algorithm:
    1. find all possible combinations of powers for all features with sum of powers less than or equal to degree
    (ex. for 3 features -> (x1,x2,x3) => combinations are (0,0,1),(0,1,0)...(1,2,0)...))

    2. take power of features in feature matrix(ie X) with corresponding combination and multiply all of the features.
    (ex. for combination (0,2,2) => powered_terms = (x1^0,x2^2,x3^2) => product of them is 1*(x1^2)*(x2^2))
    this represents a single extra feature

    3. add this generated feature into the feature matrix as new feature.

    4. repeat step 2 and 3 untill all combinations are processed.
    """

    # importing itertools functions to reduce function size
    from itertools import product, filterfalse
    # initial observations
    m, n = X.shape

    if bias_unit:
        output = np.ones([m, 1])
    else:
        output = np.array([]).reshape(m, 0)

    # product function generates all possible combinations of powers
    # the filterfalse function returns combinations whose sum of powers is <= degree
    # the np.prod function returns an array with product of all columns

    for p in filterfalse(lambda x: not (sum(x) <= degree), product(*([range(degree+1)]*n))):
        powered_terms = X**np.array([p])
        output = np.concatenate([output, np.prod(powered_terms, axis=1).reshape(m, 1)], axis=1)

    return output[:, 1:]

