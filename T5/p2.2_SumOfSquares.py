"""
Problem 2.2 - Optimization. Sum of Squares
Fitting function using Gauss-Newton to minimize the sum of squares of residuals.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt

from GaussNewton import GaussNewton


def phi(t,coeffs):
    c=96.5
    base = (1-coeffs[0]*t/coeffs[1])
    exponent = (1/(coeffs[0]*c)-1)

    return base**exponent


# Independet variable points
t = np.array([2000, 5000, 10000, 20000, 30000, 50000])

# Response data points
d = np.array([0.9427, 0.8616, 0.7384, 0.5362, 0.3739, 0.3096])

c = 96.05

init_guess = np.array([1,-1])/1000
solver = GaussNewton(phi,init_guess=init_guess)
coeffs = solver.fit(t,d,init_guess=init_guess, print_jacobian=False)
residuals = solver.get_residual()

x = np.linspace(t[0],t[-1],1000)
plt.scatter(t,d,c="r",label = "data points")
plt.plot(x,phi(x,coeffs),"k:",label="fitting")
plt.xlabel("t");plt.ylabel("d")

plt.legend()
plt.show()

# In this case, a very carefoul initial guess has to be chosen, since other whise
# fractional exponents of a negative number can be obtained, which is problematic.
# The initial guess of (1,-1)/1000 was chosen to avoid this problem.