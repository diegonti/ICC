"""
Problem 3.3.2 - Infinite sum.
Approximating exp(x) with infinite sum.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt

def factorial(x):
    result = 1
    for i in range(1,x+1): result *= i
    return result

def exponential(x):
    eps = 1
    while eps+1.0>1.0:
        eps = eps*0.5

    result, n = 0, 0

    while True:
        step = x**n/factorial(n)
        result += step 

        if step <= eps:
            Erel = abs(result-np.e)/np.e
            return result
            break
        n += 1

# Data creation
x, fx = [],[]
for i in np.arange(0,10,0.1):
    x.append(i)
    fx.append(exponential(i))

# Plot Settings
plt.plot(x,fx)
plt.xlabel("x");plt.ylabel("exp(x)")
plt.show()