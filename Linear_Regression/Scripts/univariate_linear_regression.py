"""
Example 1 - Single Variable Linear Regression

NOTE: The example and sample data is being taken from the "Machine Learning course by Andrew Ng" in Coursera.

Problem:
  Suppose you are selling your house and you
want to know what a good market price would be. One way to do this is to
first collect information on recent houses sold and make a model of housing
prices.
"""

import matplotlib.pyplot as plt
# initial imports
import numpy as np

plt.ion()

from models.linear_regression import compute_cost, gradient_descent


# ----------------Loading X and y matrix ---------------
print('Loading data ...')

data = np.loadtxt('data/ex1data1.txt', delimiter=',')
X = data[:, 0:-1]
y = data[:, -1:]
m = y.shape[0]

# printing first 5 elements
print(X[0:5])

# ----------------Plotting Data-----------------
fig = plt.figure("data")
axes1 = fig.add_subplot()
axes1.scatter(X, y, marker="x", c='r')

# ---------------Cost and Gradient descent------------
# adding bias units to X

X = np.hstack((np.ones([m, 1]), X))

# Some gradient descent settings
iterations = 1500
alpha = 0.01
theta = np.zeros([X.shape[1], 1])

print('\nTesting the cost function ...')
# compute and display initial cost
J = compute_cost(X, y, theta);
print('With theta = [0 ; 0]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = compute_cost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 54.24')



print('\nRunning Gradient Descent ...')
# run gradient descent
theta, _  = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' [-3.6303\n  1.1664]')

#---------------- plotting the linear model---------------------------
plt.plot(X[:,1:], np.dot(X,theta), lw=1)
plt.show()

#------------------ plotting the J(theta0,theta1)---------------------
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([theta0_vals.size, theta1_vals.size])

# Fill out J_vals
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2,1)
        J_vals[i,j] = compute_cost(X, y, t)


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

#--------------- Surface plot------------------
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=2, cstride=2, lw=0, cmap=cm.jet)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')

#----------------- Contour plot------------------

fig1 = plt.figure()
ax1 = fig1.add_subplot()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax1.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
ax1.set_xlabel('theta_0')
ax1.set_ylabel('theta_1')


# matplotlib interactive off to view the graph at the end of program
plt.ioff()
plt.show()


