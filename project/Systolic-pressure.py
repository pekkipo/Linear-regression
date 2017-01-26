# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel('data/mlr02.xls')
X = data.as_matrix()

# Plot the original data
# age vs pressure
plt.scatter(X[:, 1], X[:, 0])
plt.show()
# weight vs pressure
plt.scatter(X[:, 2], X[:, 0])
plt.show()


data['ones'] = 1
Y = data['X1']  # since it is a systolic blood pressure
X = data[['X2', 'X3', 'ones']]

# We ll do three linear regressions. One - with X1 as an input, one - with X3 as an input and one with both of them as inputs and compare r-squared
X2_only = data[['X2', 'ones']]
X3_only = data[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Y_hat = X.dot(w)

    d1 = Y - Y_hat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print('r2 for x2 age', get_r2(X2_only, Y))  # 0.95
print('r2 for x3 weight', get_r2(X3_only, Y))  # 0.94
print('r2 for both', get_r2(X, Y))  # best model 0.97