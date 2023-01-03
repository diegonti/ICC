"""
Problem 2.1 - Optimization. Sum of Squares
Fitting function using Gauss-Newton to minimize the sum of squares of residuals.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt

from GaussNewton import GaussNewton

# Since its used in other problems, I've created a general class
# with the GaussNewton solver that is inported in each exercise


def phi(t,coeffs):
    return coeffs[0]*np.exp(-coeffs[1]*t)


# Independet variable points
t = [-1, 0, 1, 2]

# Response data points
d = [2.7, 1, 0.4, 0.1]


init_guess = np.array([1,1])
solver = GaussNewton(phi)
coeffs = solver.fit(t,d,init_guess=init_guess, print_jacobian=True)
residuals = solver.get_residual()

x = np.linspace(t[0],t[-1],1000)

plt.plot(x,phi(x,coeffs), "k:",label = "fitting")
plt.scatter(t,d,c="r", label="data points")
plt.xlabel("t");plt.ylabel("d")

plt.legend()
plt.show()

# Given a good initial guess, the method takes ontly one iteration to converge.