### Exercise 3.2.2
##Finding root of f(x) = cos(x) - x*sin(x) = 0

import numpy as np

#Newton's Method
def newtonRaphson(f,df,xmin, digits):
    """Newton-Raphson method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Newton-Raphson Method...")

    accuracy = 10**-digits
    i = 0
    while True:
        xmax = xmin - f(xmin)/df(xmin)

        if abs(xmax-xmin) <= accuracy: 
            print(f"Root: x = {xmax:.{digits}f} ({i} iterations)")
            break

        xmin = xmax
        i += 1
 
##Main Program
#Since we are able too look at a graphical representation of this function,
#we can see that there are more than one roots, simmetryc to the y axis. So we'll begin with two different starting points, -1 and 1.
#Notice that the function has a maxima in 0, so starting from 0 or close to it may lead to non-convergence.

def f(x): return np.cos(x) - x*np.sin(x) 
def df(x): return -2*np.sin(x)-x*np.cos(x)

start1 = -1
start2 = 1
digits = 8
newtonRaphson(f,df,start1,digits=8)
newtonRaphson(f,df,start2,digits=8)