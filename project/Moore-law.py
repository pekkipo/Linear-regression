# Code Moore's law

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')
# learn about this

# Pandas couldn't read the file
for line in open('data/moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)


X = np.array(X)
Y = np.array(Y)
print(X, Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# LINEAR REGRESSION
# denominator is the same in a and b
denominator = X.dot(X) - X.mean() * X.sum()

a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean()*X.dot(X) - X.mean()*X.dot(Y) ) / denominator

# Prediction
Y_hat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Y_hat)  # line of best fit
plt.show()

# Estimate the model
d1 = Y - Y_hat  # how far we are off
d2 = Y - Y.mean()  # how far we are off if just predicted the mean
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print("a:", a, "b", b)
print("R_squared", r2)

# What really want to know is how long does it take for the transistor count to double:

# log(t_count) = a*year + b
# tc = exp(b) * exp(a*year)
# 2*tc = 2*exp(b)*exp(a*year)=exp(ln(2))*exp(b)*exp(a*year)
#   = exp(b)*exp(a*year)+ln(2)
# exp(b)*exp(a*year2) = exp(b)*exp(a*year1 + ln2)
# a*year2 = a*year1 +ln2
# year2 = year1+ln2/a
print("Time to double:", np.log(2)/a, "years")
# gives almost 2 years