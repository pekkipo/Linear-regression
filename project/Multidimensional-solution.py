import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# load the data
X = []
Y = []
# data = pd.read_csv('data/data_2d.csv', header=None)
# # Add one column filled with ones
# data['bias_term'] = 1
# print(data)
#
# X = data[['bias_term', 0, 1]]
# Y = data[2]
# With Pandas data 3D plot didn't work. Should have converted Series into a list I guess

# X = X.as_matrix() - SOLVES THE PROBLEM

for line in open('data/data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

print(X)
print(Y)

# Plot of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], Y)  # 3D plot
# ax.Axes3D.scatter(X[:, 0], X[:, 1], Y)
# X[:, 0] all rows first column
plt.show()

# Calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
# we want matrix multiplication, so we aren't using * but rather dot

Y_hat = np.dot(X, w)

# Estimate the model
d1 = Y - Y_hat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(r2)  # 0.998


