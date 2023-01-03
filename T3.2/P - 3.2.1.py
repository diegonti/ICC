### Exercise 3.2.1
##Finding root of f(x) = cos(x) - x = 0

import numpy as np

#Sequential Method
def sequential(f, xmin,xmax, digits):
    """Sequential method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Sequential Method...")

    accuracy = 10**-digits                          #Desired accuracy

    x = np.arange(xmin,xmax+accuracy, accuracy)     #x array with steps
    fx = f(x)                                       #values of f(x)
    i=0 
    while True:
        try:
            fx1 = fx[i]
            fx2 = fx[i+1]
            if fx1*fx2 <= 0:
                print(f"Root: x ∈ [{x[i]:.{digits}f}, {x[i+1]:.{digits}f}] ({i} iterations)")
                break
            i +=1 
        except IndexError: 
            print("Zero not found in the specified range.")
            break

#Dichotomy Method
def dichotomy(f,xmin,xmax, digits):
    """Dichotomy method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Dichotomy Method...")

    accuracy = 10**-digits
    i=0
    while True:
        x2 = (xmin+xmax)/2
        
        if f(x2)*f(xmax) < 0: xmin=x2
        elif f(x2)*f(xmin) < 0: xmax=x2
        else: 
            print("Zero not found in the specified range.")
            break

        if abs(xmin-xmax) <= accuracy: # If Converged
            print(f"Root: x ∈ [{xmin:.{digits}f}, {xmax:.{digits}f}] ({i} iterations)")
            break
        i+=1
        
#Secant Method
def secant(f,xmin,xmax, digits):
    """Secant method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Secant Method...")

    accuracy = 10**-digits
    i=0
    while True:
        x2 = xmin - f(xmin)*(xmax-xmin)/(f(xmax)-f(xmin))

        if abs(x2-xmax) <= accuracy: # If Converged
            print(f"Root: x = {x2:.{digits}f} ({i} iteration)")
            break
        i += 1
        xmin, xmax = xmax, x2
        
#False Position Method
def falsePosition(f,xmin,xmax, digits):
    """False position method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with False Position Method...")

    accuracy = 10**-digits
    i = 0
    while True:
        x2 = xmin - f(xmin)*(xmax-xmin)/(f(xmax)-f(xmin))

        if abs(x2-xmin) <= accuracy: # If Converged
            print(f"Root: x = {x2:.{digits}f} ({i} iterations)")
            break

        if f(x2)*f(xmax) < 0: xmin=x2
        elif f(x2)*f(xmin) < 0: xmax=x2
        else: 
            print("Zero not found in the specified range.")
            break
        i += 1

#Newton's Method
def newtonRaphson(f,df,xmin, digits):
    """Newton-Raphson method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Newton-Raphson Method...")

    accuracy = 10**-digits
    i = 0
    while True:
        xmax = xmin - f(xmin)/df(xmin)

        if abs(xmax-xmin) <= accuracy: # If Converged
            print(f"Root: x = {xmax:.{digits}f} ({i} iterations)")
            break

        xmin = xmax
        i += 1
        
#Mullers Method
def muller(f,x0,x1,x2, digits):
    """Muller's method for finding roots.\n\nTakes a function f, the range to find the root, and the digit accuracy."""
    print("Searching root with Muller's Method...")

    i=0
    accuracy = 10**-digits
    x0,x1,x2 = complex(x0), complex(x1), complex(x2)
    while True:
        y0,y1,y2 = f(x0),f(x1),f(x2)
        
        q = (x2-x1)/(x1-x0)
        A = q*y2 - q*(1+q)*y1 + q**2*y0
        B = (2*q+1)*y2 - (1+q)**2*y1 + q**2*y0
        C = (1+q)*y2
        radical = np.sqrt(B**2 - 4*A*C)

        x3 = x2 - (x2-x1)*2*C/(max(B+radical, B-radical))

        if abs(x3-x2) <= accuracy: # If Converged
            print(f"Root: x = {x3:.{digits}f} ({i} iterations)")
            break

        x0 = x1
        x1 = x2
        x2 = x3

        i += 1


##Main Program
#Since we are able too look at a graphical representation of this function,
#we can see that the root is between 0.5 and 1, so this will be the search interval.

def f(x): return np.cos(x) - x 
def df(x): return -np.sin(x) - 1 

xmin,xmax = 0.5,1
digits = 8
sequential(f,xmin,xmax,digits=8)
dichotomy(f,xmin,xmax,digits=8)
secant(f,xmin,xmax,digits=8)
falsePosition(f,xmin,xmax,digits=8)
newtonRaphson(f,df,xmax,digits=8)
muller(f,0,xmin,xmax,digits=8)