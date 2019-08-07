"""
Example 1 - Single Variable Linear Regression

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
import matplotlib as mp
import seaborn as sp
from models.data_preprocessing import feature_normalize


# ----------------Loading X and y matrix ---------------
print('Loading data ...')

data = np.loadtxt('data/linear_reg/ex1data2.txt')
X = data[:, 1:2]
y = data[:, 3]
m = y.length

# printing first 5 elements
print(X[0:5, :])

# ----------------Feature Normalization -----------------
print("Normalizing the features")
X, mu, sigma = feature_normalize(X)


