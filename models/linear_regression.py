"""
 Linear Regression Model

 This file contains functions to implement single and multi variable linear regression on the given data set.
"""
from typing import List, Any

import numpy as np
# importing scipy.optimize.minimize for training linear regression
from scipy.optimize import minimize


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
    theta_cpy = theta.copy()

    for i in range(num_iters):
        h = X.dot(theta_cpy)
        theta_cpy -= (alpha / m) * X.T.dot((h - y))
        # theta -= (alpha / m) * np.dot(X.T, (np.dot(X, theta)-y))
        J_history[i] = compute_cost(X, y, theta_cpy)

    # theta -= (alpha/m).*(X' *((X*theta)-y));

    return theta_cpy, J_history


def compute_cost(X, y, theta, lamda=0):
    """
    computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    :param X: numpy array of shape (m,n)
        Training data
    :param y: matrix(m,)
        Target values corresponding to X.
    :param theta: matrix(n,)
        Weights
    :param lamda: float
        regularizes the cost function according to lamda
    :return: float value
        Cost for given training data nad weights.
    """
    m = y.size  # training data size.
    if theta.ndim < 2:
        theta = theta[:, np.newaxis]
    if y.ndim < 2:
        y = y[:, np.newaxis]

    predictions = np.dot(X, theta)  # h(X)

    J = np.dot((predictions - y).T, (predictions - y)) / (2 * m)  # cost over the training data
    if lamda:
        J += (lamda / (2 * m)) * np.sum(np.square(theta[1:, :]))

    return J.ravel()


def compute_gradient(X, y, theta, lamda=0):
    """
    computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    :param X: numpy array of shape (m,n)
        Training data
    :param y: matrix(m,)
        Target values corresponding to X.
    :param theta: matrix(n,)
        Weights
    :param lamda: float
        regularizes the gradient according to lamda
    :return grad: matrix(n,)
        gradient for given training data and weights.
    """
    m = X.shape[0]
    if theta.ndim < 2:
        theta = theta[:, np.newaxis]
    if y.ndim < 2:
        y = y[:, np.newaxis]

    h = np.array(X @ theta)
    grad = (1 / m) * (X.T @ (h - y))
    if lamda:
        grad[1:] += (lamda / m) * theta[1:]
    return grad


def normal_equation(X, y):
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

    theta = np.dot(np.linalg.inv(X.T.dot(X)), X.T.dot(y))

    return theta


# callback function to record costs after each iteration
def record_cost(lst, cost):
    lst.append(cost)  # list append method implements in place adition


def train_linear_reg(X, y, lamda, method="TNC"):
    # creating short hand for the cost function
    cost_fun = lambda p: compute_cost(X, y, p, lamda)

    # creating short hand for the gradient function
    grad_fun = lambda p: compute_gradient(X, y, p, lamda)

    cost_list = []  # this list will store cost after each iteration

    # creating short hand for the record_cost function
    rec_cost = lambda p: record_cost(cost_list, p)

    # initializing thetas
    initial_theta = np.zeros([X.shape[1]])

    result = minimize(fun=cost_fun, x0=initial_theta, jac=grad_fun, method=method, callback=rec_cost)

    # print("Output: \n{}".format(result))
    return result, cost_list


def validation_curve(X, y, X_val, y_val, lamda_vec=None, method="TNC"):
    if lamda_vec is None:
        lamda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    error_train = []
    error_val = []

    for lamda in lamda_vec:
        temp_result, _ = train_linear_reg(X, y, lamda, method)
        error_train.append(compute_cost(X, y, temp_result.x, 0))
        error_val.append(compute_cost(X_val, y_val, temp_result.x, 0))

    return lamda_vec, error_train, error_val
