"""
Problem 3.4.1 - Interpolation.
Interpolating function from datapoints.
Diego
"""
import numpy as np
import matplotlib.pyplot as plt


def interpolate(xPoints,yPoints,x):
    """Interpolates x (number or array) linearly to a data table"""

    # If x is number
    if type(x) == int or type(x) == float: 
        for i,p in enumerate(yPoints):
            try: # Try-Except in case the index is out of bounds (for the las number)
                # Get the table points interval
                x1,x2 = xPoints[i],xPoints[i+1]
                fx1,fx2 = yPoints[i],yPoints[i+1]
                
                # Calculate the linear interpolation in the interval
                if x >= xPoints[i] and x < xPoints[i+1]:
                    fx = fx1 + (x-x1)*(fx2-fx1)/(x2-x1)
                    return fx
            except IndexError: pass

    # If x is an iterable (array)        
    else: 
        fx = np.array([])
        for j in x:
            for i,p in enumerate(yPoints):
                try:
                    x1,x2 = xPoints[i],xPoints[i+1]
                    fx1,fx2 = yPoints[i],yPoints[i+1]

                    if j >= xPoints[i] and j <= xPoints[i+1]:
                        fxj = fx1 + (j-x1)*(fx2-fx1)/(x2-x1)
                        fx=np.append(fx,[fxj])
                        break
                except IndexError: pass
        return fx

###################  Main Program  #################
def f(x): return np.cos(x)

# Original Function
x100 = np.linspace(0,10, num = 100)
x1000 = np.linspace(0,10, num = 1000)
fx = f(x1000)

# Table Points
xPoints = np.linspace(0,10,20)
yPoints = f(xPoints) 

# Plot Settings
plt.plot(x1000,fx)
plt.plot(xPoints,yPoints,"x")
plt.plot(x100,interpolate(xPoints,yPoints,x100))
plt.xlabel("x");plt.ylabel("y")
plt.legend(["cos(x)", "table points", "interpolation"])
plt.show()


            
