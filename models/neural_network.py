import numpy as np
from .data_preprocessing import add_bias_unit
from .logistic_regression import sigmoid


def sigmoid_gradient(z):
    """
    Computes the gradient of sigmoid function evaluated at z.

    :param z: z can be a matrix, vector or scalar
    :return: numpy matrix same as size of z
        sigmoid values of z
    """
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    """
    Computes the cost and gradient of the neural network with 1 hidden layer of
    given parameters. The parameters for the neural network are "unrolled" into
    the vector nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.

    Note: This layer is capable of working with only 3 layers.

    :param nn_params: numpy array of one dimension
        unrolled neural network parameter
    :param input_layer_size: int
        number of nodes in input layer
    :param hidden_layer_size: int
        number of nodes in hidden layer
    :param num_labels:
        number of output layer nodes
    :param X: numpy array of shape (m,n)
        Training data
    :param y: numpy array of shape (m,1)
        Training predictions
    :param lamda: Floating point value
        Regularization parameter
    :return: J
        cost of the neural network
    """

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    theta1 = nn_params[:(hidden_layer_size * (input_layer_size + 1))].reshape([hidden_layer_size, (input_layer_size + 1)])
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape([num_labels, (hidden_layer_size + 1)])

    m = X.shape[0]

    #   Part 1: Feedforward the neural network and return the cost in the
    #         variable J.
    #
    #   Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively.
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #   Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.

    # adding bias unit to X
    a1 = add_bias_unit(X)

    # Part 1: Feed propagation to get J
    # layer 2
    a2 = sigmoid(a1 @ theta1.T)
    a2 = add_bias_unit(a2)

    # layer 3
    a3 = sigmoid(a2 @ theta2.T)

    # converting y into vector form i.e. binary columns for each class
    # y_matrix = np.eye(num_labels)[y.reshape(-1)]
    y_bin = (y == (np.arange(num_labels) * np.ones([m, num_labels]))).astype(np.int)

    h = a3  # for formula convenience
    J = (-1 / m) * (np.ones([1, m]) @ ((y_bin * np.log(h)) + ((1 - y_bin) * np.log(1 - h))) @ np.ones([num_labels, 1]))
    J += (lamda / (2 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

    # Part 2: Back propagation

    d3 = a3 - y_bin
    d2 = (d3 @ theta2)[:, 1:] * sigmoid_gradient(a1 @ theta1.T)
    # d2 = d2[:, 1:]

    theta2_grad = ((1 / m) * (d3.T @ a2))
    theta2_grad[:, 1:] += (lamda / m) * theta2[:, 1:]

    theta1_grad = ((1 / m) * (d2.T @ a1))
    theta1_grad[:, 1:] += (lamda / m) * theta1[:, 1:]

    # unroll gradient
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()], axis=0)

    return J, grad


def rand_initialize_weights(c_out, c_in, debug=False):
    """
    Initialize the weights of a layer with c_in incoming connections and
    c_out outgoings connections.

    In case of debug=True the weights are initialized using a fixed strategy.
     This will help you later in debugging.

    :param c_in: incoming connections
    :param c_out: outgoings connections
    :param debug: boolean to initlialize weights for debugging
    :return: numpy matrix with randomly initialized weights
    """

    if debug:
        w = np.sin(np.arange(1, 1 + (1 + c_in) * c_out)) / 10.0
        w = w.reshape(c_out, 1 + c_in)
    else:
        epsilon = 0.12
        w = np.random.rand(c_out, 1 + c_in) * 2 * epsilon - epsilon
        # w = np.random.uniform(-epsilon, epsilon, [c_out, c_in + 1])
    return w


def numerical_gradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.

    Calling y = J(theta) should return the function value at theta.

    Notes: The following code implements numerical gradient checking, and
        returns the numerical gradient.It sets numgrad(i) to (a numerical
        approximation of) the partial derivative of J with respect to the
        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
        be the (approximately) the partial derivative of J with respect
        to theta(i).)


    :param J : func
        The cost function which will be used to estimate its numerical gradient.
    :param theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.
    :param e : float (optional)
        The value to use for epsilon for computing the finite difference.
    :return: numpy array same as that of size of theta

    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1) / (2.0 * e)
    return numgrad


def check_nn_gradients(cost_function, lamda=0):
    """Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.

    :param lamda : float (optional)
        The regularization parameter value.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = rand_initialize_weights(hidden_layer_size, input_layer_size, debug=True)
    Theta2 = rand_initialize_weights(num_labels, hidden_layer_size, debug=True)

    # Reusing debugInitializeWeights to generate X
    X = rand_initialize_weights(m, input_layer_size - 1)
    y = np.arange(1, 1 + m).reshape([-1, 1]) % num_labels
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([Theta1.flatten(), Theta2.flatten()], axis=0)

    # short hand for cost function
    costFunc = lambda p: cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    cost, grad = costFunc(nn_params)
    numgrad = numerical_gradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)
