"""
Problem 3.4.1 - Interpolation.
Interpolating function from datapoints with Lagrange method.
Diego
"""
import numpy as np
import matplotlib.pyplot as plt

def lagrange(x,xi,xPoits):
    """Calulates Lagrange coeffitient for a given xi in a table of xPoints."""
    result = 1
    for xj in xPoints:
        if xi != xj: result *= (x-xj)/(xi-xj)
    return result

def interpolate(xPoints,yPoints,x):
    """Lagrange Polynomial Interpolation of a value x in a grid of x and y points."""
    p = 0 
    for i,xi in enumerate(xPoints):
        p += lagrange(x,xi,xPoints) * yPoints[i]
    return p

###################  Main Program  #################
def f(x): return np.cos(x)

# Number o points and step
n, h = 20,0.1                           # n+1 points --> n degree
x = np.arange(0,n*h+h,h)                # For interpolation
xSmooth = np.linspace(0,n*h+h,1000)     # For representing cos
fx = f(xSmooth)

# Table points (grid)
xPoints = np.linspace(0,n*h+h,n)
yPoints = f(xPoints)

# Interpolation
yInterpolate = []
for xi in x: yInterpolate.append(interpolate(xPoints,yPoints,xi))

# Plot Settings
plt.plot(xSmooth,fx)
plt.plot(xPoints,yPoints, "x")
plt.plot(x,yInterpolate)

plt.legend(["cos(x)", "table points", "interpolation"])
plt.show()

# One can see that the Lagrange polynomial interpolation does a good job of recreating the function at
# higher n. Nevertheless, at very high n's (<70) the interpolation starts to break.