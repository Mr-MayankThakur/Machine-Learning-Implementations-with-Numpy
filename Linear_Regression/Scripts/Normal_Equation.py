"""
Example 3 - Linear Regression using Normal Equation

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

import numpy as np
from models.data_preprocessing import add_bias_unit
from models.linear_regression import normal_equation


