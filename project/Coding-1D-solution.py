# See notes for solution

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data
# X = []
# Y = []
# I ll use pandas as it returns numpy arrays
data = pd.read_csv("data/data_1d.csv", header=None)  # consult Pandas docs for arguments
print(data)
X = data[0]  # in pandas its a first column
Y = data[1]

# Plot this data
plt.scatter(X, Y)
plt.show()

# denominator is the same in a and b
denominator = X.dot(X) - X.mean() * X.sum()
# X.dot(X) is a sum of Xi squared in the formula

a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
# dot because there is a sum of YiXi in the formula
b = ( Y.mean()*X.dot(X) - X.mean()*X.dot(Y) ) / denominator

# Calculate the predicted Y
Y_hat = a*X + b

# plot it all
plt.scatter(X, Y)  # data
plt.plot(X, Y_hat) # line of best fit
plt.show()


# ESTIMATE THE MODEL.
# R-squared

d1 = Y - Y_hat
d2 = Y - Y.mean() # mean is a scalar but numpy will subtract it from all Ys
r2 = 1 - d1.dot(d1)/d2.dot(d2)
# We want to square the differences. For that we calculate the dot product treating each i as an index of a vector

print("R-squared is {}".format(r2))
# gives r-squared 0.99. Close to 1. Very good model

# ADDING MORE DIMENSIONS IMPROVES R SQUARED! If is those dimensions are not useful at all