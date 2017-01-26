
import numpy as np
import matplotlib.pyplot as plt

N = 10  # number of points
D = 3  # dimensionality
X = np.zeros((N, D))
X[:,0] = 1  # bias term
X[:5, 1] = 1  # set first 5 elements of 1st column to one
X[5:, 2] = 1  # set last elements of second column to one
Y = np.array([0]*5 + [1]*5)  # set the targets. first half zeros, second half ones
# [0,0,0,0,0,1,1,1,1,1]

# print X so you know what it looks like
print("X:", X)

# won't work! Because cannot inverse singular matrix
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# try gradient descent
costs = []  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D)  # randomly initialize w. Initialize random weights
learning_rate = 0.001
for t in range(1000):  # 1000 epochs
    # update w
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate*X.T.dot(delta)  # formula from the notes

    # find and store the cost in costs array
    # mean square error
    mse = delta.dot(delta) / N
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()
#  cost drops in every iteration

print("final w:", w)

# plot prediction vs target
plt.plot(Y_hat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()