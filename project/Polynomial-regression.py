# Linear regression can be applied to a polynomial
# Polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/data_poly.csv', header=None)
data['bias_term'] = 1
data['x*x'] = data.apply(lambda x: x[0]*x[0], axis=1)
# one column is just squared x
#print(data)

# X - inputs
# Y - outputs

X = data[['bias_term', 0, 'x*x']]
Y = data[1]

X = X.as_matrix()
# can convert to numpy array instead of a dataframe. Might be needed for 3d scatter later on
# YES. it was needed
print(X)

# Plot
plt.scatter(X[:, 1], Y)
plt.show()

# Calculate the weights
# just a multiple regression. Difference is only in a way we created the input X

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# Plot it all together
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))  # predicted y as a line
# sorted we need because points might be not in order and this will lead to a very strange picture
plt.show()


