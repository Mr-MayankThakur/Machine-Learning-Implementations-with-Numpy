"""
 Linear Regression Model

 This file contains functions to implement single and multi variable linear regression on the given data set.
"""
import numpy as np


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    gradient_descent performs gradient descent on the data to learn thetas.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    :param X: numpy array of shape (m,n)
        Training data
    :param y: matrix(m,1)
        Target values corresponding to X.
    :param theta: matrix(n,1)
        Initial Weights
    :param alpha: floating point value
        Learning rate
    :param num_iters: integer value
        The number of times the gradient descent should run.
    :return: theta, J_history
        thetas are the updated thetas after gradient descent.
        J_history is the cost upon each iteration.
    """

    # initializing some variables for future use
    m = X.shape[0]  # number of training examples
    J_history = np.zeros([num_iters, 1])

    for i in range(num_iters):
        theta -= (alpha / m) * np.dot(X.T, (np.dot(X, theta)-y));
        J_history[i] = compute_cost(X, y, theta)

    # theta -= (alpha/m).*(X' *((X*theta)-y));

    return theta, J_history


def compute_cost(X, y, theta):
    """
    computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    :param X: numpy array of shape (m,n)
        Training data
    :param y: matrix(m,1)
        Target values corresponding to X.
    :param theta: matrix(n,1)
        Weights
    :return: float value
        Cost for given training data nad weights.
    """
    m = y.size  # training data size.

    predictions = np.dot(X, theta)  # h(X)

    J = np.dot((predictions - y).T, (predictions - y))/(2*m)  # cost over the training data

    return J


def normal_equation(X,y):
    """
    Computes the closed-form solution to linear
    regression using the normal equations.

    :param X: numpy array of shape (m,n)
        Training data
    :param y: matrix(m,1)
        Target values corresponding to X.
    :return: theta matrix(n,1)
        Computed Weights using normal equation
    """

    m = y.size  # training data size

    # matlab implementation -> theta = (pinv(X' * X)) * (X' * y)

    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return theta

