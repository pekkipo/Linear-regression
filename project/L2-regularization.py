import numpy as np
import matplotlib.pyplot as plt

N = 50
# Generate the date
X = np.linspace(0, 10, N)  # 50 evenly space points between 0 and 10
Y = 0.5*X + np.random.randn(N)  # + some random noise

# Manually set outliers
Y[-1] += 30  # [-1] is a last point. 30 bigger than it is in generated data
Y[-2] += 30  # penultimate points

plt.scatter(X, Y)
plt.show()

# Solve for the best weights. Add the bias term
X = np.vstack([np.ones(N), X]).T
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html

# Calculate the maximum likelihood solution. Without large weight penalty for now. The one done previously
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# predictions for w_ml
Y_hat_ml = X.dot(w_ml)
plt.scatter(X[:, 1], Y)  # with the original data
plt.plot(X[:, 1], Y_hat_ml)  # max likelihood line
plt.show()
# because of two outliers in the end the line will be pulled towards them and the line of fit isn't that good


# L2-regularization solution. With penalty
l2 = 1000.0  # penalty for the large weight
w_MAP = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
# l2*np.eye(2) is lambda*Identity_matrix from the notes
# Predictions
Y_hat_MAP = X.dot(w_MAP)
plt.scatter(X[:, 1], Y)  # with the original data
plt.plot(X[:, 1], Y_hat_ml, label='max likelihood')  # max likelihood line for reference
plt.plot(X[:, 1], Y_hat_MAP, label='MAP')  # Maximum aposteriori line
plt.legend()  # so that labels can be seen
plt.show()
# MAP solution shows the trend much better
