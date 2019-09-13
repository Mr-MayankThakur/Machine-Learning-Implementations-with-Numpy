import numpy as np


def sigmoid(z):
    """
    Computes the sigmoid of z.

    :param z: z can be a matrix, vector or scalar
    :return: numpy matrix same as size of z
        sigmoid values of z
    """
    g = 1.0/(np.exp(-1*z)+1)
    return g


def predict(X, theta, threshold=0.5):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters theta.
    Computes the predictions for X using a
    threshold (i.e., if sigmoid(theta'*x) >= threshold, predict 1)

    :param X: numpy array of shape (m,n)
        Training data
    :param theta: numpy array (n,1)
        Weights
    :param threshold: float value
        threshold value above which the the predictor should return 1
    :return: y_predicted = numpy array (m,1)
        numpy array with predicted values in terms of 0 and 1
    """

    p = sigmoid(X.dot(theta))
    return (p >= threshold).astype(np.int64)


def cost_function(theta, X, y, lamda=0.01, regularized=False):
    """
    Compute cost and gradient for logistic regression with and without regularization.

    Computes the cost of using theta as the parameter for regularized logistic regression
    and the gradient of the cost w.r.t. to the parameters.

    using lamda instead of lambda because of keyword conflict.

    :param X: numpy array of shape (m,n)
        Training data
    :param theta: numpy array (n,1)
        Weights
    :param y: numpy array of shape (m,1)
        Training predictions
    :param lamda: Floating point value
        Regularization parameter
    :param regularized: Bool(Default:True)
        if True the cost function returned will be regularized
    :return J, Grad:
        J: Cost of the theta values for given dataset
        Grad: gradient for logistic regression with regularization
        partial derivatives of the cost w.r.t. each parameter in theta
    """

    # initial values
    m = y.size
    if type(theta) != type(np.array([])):
        theta = np.array(theta).reshape([-1, 1])
    # since in regularization we do not penalize theta(0)
    #print("Message: theta = {}".format(theta))
    h = sigmoid(X @ theta)

    J = (-(y.T @ np.log(h)) - ((1-y.T) @ np.log(1-h)))/m

    if regularized:
        J = J + ((theta[1:].T @ theta[1:]) * (lamda/(2*m))) # regularization value addted to cost;
        # note we didn't add regularization for first theta

    return J


def gradient_function(theta, X, y, lamda=0.01, regularized=False):
    """
        Compute gradient for logistic regression with and without regularization.

        using lamda instead of lambda because of keyword conflict.

        :param X: numpy array of shape (m,n)
            Training data
        :param theta: numpy array (n,1)
            Weights
        :param y: numpy array of shape (m,1)
            Training predictions
        :param lamda: Floating point value
            Regularization parameter
        :param regularized: Bool(Default:True)
            if True the cost function returned will be regularized
        :return Grad:
            Grad: gradient for logistic regression with regularization
            partial derivatives of the cost w.r.t. each parameter in theta
    """
    # initial values
    m = y.size
    theta = np.array(theta).reshape([-1, 1])

    # since in regularization we do not penalize theta(0)
    h = sigmoid(X.dot(theta))

    grad = (1/m) * (X.T.dot(h-y))

    if regularized:
        # note we didn't add regularization for first theta
        # regularization term for all other thetas except 0
        grad[1:] = (((1 / m) * (X.T.dot(h - y)))[1:] + (theta[1:] * (lamda / m)))
    return grad.ravel()


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
        h = sigmoid(X.dot(theta))
        theta = theta - (alpha * gradient_function(X, y, theta))
        # theta -= (alpha / m) * np.dot(X.T, (np.dot(X, theta)-y))
        J_history[i] = cost_function(X, y, theta)

    # theta -= (alpha/m).*(X' *((X*theta)-y));

    return theta, J_history
