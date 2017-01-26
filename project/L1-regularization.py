#  The idea is to generate the fat matrix with lots of features
#  Use L1 regularization to see if we can find a sparse set of weights that identifies the useful dimensions of x

import numpy as np
import matplotlib.pyplot as plt

# Fat matrix
N = 50  # number of points
D = 50  # number of dimensions

# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5)*10
#  centered around zero from -5 to 5

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))  # last D - 3 (47)  are zeroes and do not influence the output

# generate Y - add noise with variance 0.5
Y = X.dot(true_w) + np.random.randn(N)*0.5  # + gaussian random noise

# perform gradient descent to find w
costs = []  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0  # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
for t in range(500):
    # update w
    Y_hat = X.dot(w)  # prediction
    delta = Y_hat - Y
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    # find and store the cost
    mse = delta.dot(delta) / N
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot our w vs true w
plt.plot(true_w, label='true w')
plt.plot(w, label='w_MAP')
plt.legend()
plt.show()