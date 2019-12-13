"""
Example 2 - Multi Variable Linear Regression

NOTE: The example and sample data is being taken from the "Machine Learning course by Andrew Ng" in Coursera.

Problem:
  Suppose you are the CEO of a restaurant franchise and are considering
  different cities for opening a new outlet. The chain already has trucks
  in various cities and you have data for profits and populations from
  the cities. You would like to use this data to help you select which
  city to expand to next.

  The file 'data/linear_reg/ex1data1.txt' contains the dataset for our
  linear regression problem. The first column is the population of a city
  and the second column is the profit of a food truck in that city.
  A negative value for profit indicates a loss.
"""

# initial imports
import numpy as np
import matplotlib.pyplot as plt
from models.data_preprocessing import feature_normalize, add_bias_unit
from models.linear_regression import gradient_descent, normal_equation
plt.ion()

# ----------------Loading X and y matrix ---------------
print('Loading data ...')

data = np.loadtxt('data/ex1data2.txt', delimiter=',')
X = data[:, :-1]  # 47x2
y = data[:, -1, None]  # 47x1
m = y.size  # 47

# printing first 5 elements
print(X[0:5, :])

# ----------------Feature Normalization -----------------
print("Normalizing the features")
X, mu, sigma = feature_normalize(X)

# adding intercept term to X
X = add_bias_unit(X)

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.03
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros((X.shape[1], 1))
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
fig = plt.figure("Covariance Graph")
ax = fig.subplots()
ax.plot(range(J_history.size), J_history, lw=2)
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Cost J')
fig.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house

test_data = np.array([1650,3]).reshape(1, 2)
test_data, _, __ = feature_normalize(test_data, mu, sigma)
test_data = add_bias_unit(test_data)
price = test_data.dot(theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {}'.format(price))

# ================Normal Equations ================

print('Solving with normal equations...')

# ----------------Loading X and y matrix ---------------
print('Loading data ...')

data = np.loadtxt('data/ex1data2.txt', delimiter=',')
X = data[:, :-1]  # 47x2
y = data[:, -1, None]  # 47x1
m = y.size  # 47


# Add intercept term to X
X = add_bias_unit(X)

# Calculate the parameters from the normal equation
theta = normal_equation(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: ')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
test_data = np.array([1650,3]).reshape(1, 2)
test_data = add_bias_unit(test_data)
price = test_data.dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {}'.format(price))

# bloking matplotlib figures for obervations
plt.ioff()
plt.show()