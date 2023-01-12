"""
Problem 5.3.3 - Optimization. Sum of Squares
Optimizing parameters x1 and x2 by minimizing given f.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt

def phi(t,coeffs):
    return coeffs[0]*np.exp(-coeffs[1]*t)

def x1(x2,d,t):
    return np.sum(d*np.exp(-x2*t))/np.sum(np.exp(-2*x2*t))

def f(x1,x2,d,t):
    return np.sum(d**2 -x1*np.sum(d*np.exp(-x2*t)))


# Independet variable points
t = np.array([-1, 0, 1, 2])

# Response data points
d = np.array([2.7, 1, 0.4, 0.1])

x2_values = np.linspace(0,2,1000)

x1_values,f_values = [],[]
for x2i in x2_values:
    x1i = x1(x2i,d,t)
    fi = f(x1i,x2i,d,t)
    x1_values.append(x1i)
    f_values.append(fi)

x1_values,f_values = np.array(x1_values), np.array(f_values)
fmin = np.min(f_values)
fmin_loc = np.argmin(f_values)

coeffs = [x1_values[fmin_loc],x2_values[fmin_loc]]
print("Optimized parameters: ",*coeffs)


x = np.linspace(t[0],t[-1],1000)

plt.plot(x,phi(x,coeffs), "k:",label = "fitting")
plt.scatter(t,d,c="r", label="data points")
plt.xlabel("t");plt.ylabel("d")

plt.legend()
plt.show()

# Using this method, converges in almos the same values obtained in problem 1.