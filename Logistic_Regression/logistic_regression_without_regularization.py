"""
Example 3 - Logistic Regression without regularization

NOTE: The example and sample data is being taken from the "Machine Learning course by Andrew Ng" in Coursera.

Problem:
  Suppose that you are the administrator of a university department and
  you want to determine each applicant’s chance of admission based on their
  results on two exams. You have historical data from previous applicants
  that you can use as a training set for logistic regression. For each training
  example, you have the applicant’s scores on two exams and the admissions
  decision.
  Your task is to build a classification model that estimates an applicant’s
  probability of admission based the scores from those two exams.

  The file 'data/ex1data1.txt' contains the dataset for our
  Logistic regression problem.
"""

#initial imports

import numpy as np
from matplotlib import pyplot as plt
plt.ion()
from models.data_preprocessing import add_bias_unit, map_feature, feature_normalize
from models.logistic_regression import cost_function, predict, gradient_descent, gradient_function, sigmoid
from models.plotter import plot_decision_boundary
data = np.loadtxt('data/ex2data1.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1, np.newaxis]

"""
==================== Part 1: Plotting ====================
  We start the exercise by first plotting the data to understand the 
  the problem we are working with.
"""
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
"""
Example plotting for multiple markers
x = np.array([1,2,3,4,5,6])
y = np.array([1,3,4,5,6,7])
m = np.array(['o','+','+','o','x','+'])

unique_markers = set(m)  # or yo can use: np.unique(m)

for um in unique_markers:
    mask = m == um 
    # mask is now an array of booleans that van be used for indexing  
    plt.scatter(x[mask], y[mask], marker=um)
"""
fig, ax = plt.subplots()
y_slim = y.ravel()
# plotting y=1 values
ax.scatter(x=X[y_slim == 1, 0], y=X[y_slim == 1, 1], marker='+', color='black', s=50)
# plotting y=0 values
# X[y_slim == 0, 0] is logical indexing with rows with y=0 only
ax.scatter(x=X[y_slim == 0, 0], y=X[y_slim == 0, 1], marker='o', color='y', s=25)

# labels
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')

# Specified in plot order
ax.legend(['Admitted', 'Not admitted'])


# ============ Part 2: Compute Cost and Gradient ============
# initial sizes
m, n = X.shape

# adding bias unit
X = add_bias_unit(X)


# Initialize fitting parameters
initial_theta = np.zeros([n + 1, 1])

# Compute and display initial cost and gradient
cost = cost_function(initial_theta, X, y, regularized=False)
grad = gradient_function(initial_theta, X, y, regularized=False)
print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')


# =========== Performing gradient descent================
#from models.data_preprocessing import feature_normalize
X_norm, mu, sigma = feature_normalize(X[:, 1:])
X_norm = add_bias_unit(X_norm)

from scipy.optimize import minimize

#theta_history = np.array([]).reshape([0, n+1])
theta_history = []


def cg(abc, *args):
    theta_history.append(abc)


initial_theta = np.zeros(n+1)
op_result = minimize(fun=cost_function, x0=initial_theta, jac=gradient_function, args=(X, y, 0.01, False), method='cg', callback=cg)


#cost = cost_function(op_result.x,X,y, regularized=False)
print('Cost at theta found by Gradient descent: {}'.format(op_result.fun))
print('Expected cost (approx): 0.203')
print('theta: {}'.format(op_result.x))
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# converting theta_history into J_history
J_history = (np.array(theta_history[::-1]) @ op_result.x)

# plot J_history
fig1, ax1 = plt.subplots()
ax1.plot(range(J_history.size), J_history)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')

theta = op_result.x[:,np.newaxis]
plot_decision_boundary(theta, X, y, sigmoid, 0.1,fig, ax)

# ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, we are going to use it to predict the outcomes
#  on unseen data. In this part, we will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, we will compute the training and test set accuracies of
#  our model.

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with scores 45 and 85, we predict an admission probability of {}'.format(prob))
print('Expected value: 0.775 +/- 0.002')

# Compute accuracy on our training set
p = predict(X,theta)

print('Train Accuracy: {}'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0\n')



# ===============End================
plt.ioff()
plt.show()