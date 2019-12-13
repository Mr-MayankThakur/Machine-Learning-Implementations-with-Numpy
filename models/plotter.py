import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(theta, X, y, hypothesis, precision=0.1, fig=None,  ax=None, feature_map=None):
    """
    Plots the data points X and y into a new figure with
    the decision boundary defined by theta

    Plots the data points with + for the
    positive examples and o for the negative examples.
     X is assumed to be a either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones

    :param X: numpy array of shape (m,n)
        Training data
    :param theta: numpy array (n,1)
        Weights
    :param y: numpy array of shape (m,1)
        Training predictions
    :param h: hypothesis function
        reference to the function which will be used for cost calculation as callback
    :param fig: matplotlib figure object
        specify the figure on which you want to plot decision boundary
    :param ax: matplotlib axes object
        specify the axes on which you want to plot decision boundary
    :param fearure_map: tuple of callable function and args which should be passed to the function with training data
        if specified training data is passed throught the function before cost estimation
    :return: matplotlib figure and axes
        corresponding figure and axes on which the graph is plotted
    """

    # initializing figure and subplot
    if (ax is None) or (fig is None):
        fig, ax = plt.subplots()
        # plotting the data using matplotlib
        ax.scatter(X[(y == 1).ravel(), 1], X[(y == 1).ravel(), 2], marker='+', c='r')
        ax.scatter(X[(y == 0).ravel(), 1], X[(y == 0).ravel(), 2], marker='o', c='xkcd:light yellow', edgecolors='black')

    if X.shape[1] <= 2:

        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:, 2])-2,  max(X[:, 2])+2]

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*((theta[1]*plot_x) + theta[0])

        # Plot, and adjust axes for better viewing
        ax.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        ax.legend('Admitted', 'Not admitted', 'Decision Boundary')
        ax.axis([30, 100, 30, 100])
    else:

        # Plotting decision regions
        x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, precision), np.arange(y_min, y_max, precision))

        # Evaluate z = theta*x over the grid
        # convert the grid into list of cells in grid for passing into hypothesis function
        X = np.concatenate((np.ones((xx.shape[0]*xx.shape[1], 1)),  np.c_[xx.ravel(), yy.ravel()]), axis = 1)
        if feature_map is not None:
            X = feature_map[0](X[:,1:], *feature_map[1:])
        z = hypothesis(X.dot(theta))
        z = z.reshape(xx.shape)
        # plotting the contour with only one level
        ax.contour(xx, yy, z, levels=1, colors='blue', linewidths=1)
    return fig, ax

