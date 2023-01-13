###xercise 3.3.4 (solve with fixed-point method)
"""
Problem 3.3.4 - Solving equations.
Solving equation with the fixed point method and Aitken extrapolation.
Diego Ontiveros
"""

def fixedPoint(f,start,eps=None,maxiter=1e5):
    
    if eps == None: # Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    # Method
    x,i=start,0
    while True:
        
        step = f(x)
        eabs = abs(x-step)
        x=step
        # print(x)
        
        if eabs < eps:
            print(f"Root with Fixed Point: {x}. Iterations: {i}, Eabs={eabs}")
            break
        elif i > maxiter:
            print("Fixed Point: Max interations surpassed. Does not converge.")
            print(f"The number until now is: {x}")
            break
        i += 1

def aitken(f,start, eps=None, maxiter=1e4):
    
    if eps == None: # Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    x0,i = start,0
    while True: 
        
        x1 = f(x0)
        x2 = f(x1)

        denominator = (x2 - x1) - (x1 - x0)
        aitkenX = x2 - ((x2 - x1)**2)/denominator
        x0 = aitkenX

        eabs = abs(aitkenX-x2)
        if eabs < eps:
            print(f"Root with Aitken: {aitkenX}. Iterations: {i}, Eabs={eabs}")
            break
        elif i > maxiter:
            print("Aitken: Max interations surpassed. Does not converge.")
            print(f"The number until now is: {aitkenX}")
            break

        i += 1


import numpy as np

def g(x): return (x+1)/np.sin(3*x) + 1
def h(x): return -1 + (x-1)*np.sin(3*x)
def f(x): return np.arcsin((x+1)/(x-1))/3

print("\nUsing g(x)...")
fixedPoint(g, -0.22, eps=1e-6)
aitken(g,-0.22, eps=1e-4)
print("\nUsing h(x)...")
fixedPoint(h, -0.22, eps=1e-6)
aitken(h,-0.22, eps=1e-6)
print("\nUsing f(x)...")
fixedPoint(f, -0.22, eps=1e-6)
aitken(f,-0.22, eps=1e-6)

# One can see that using the Aitken aproximation leads to a convergence in more cases,
# being a better case for using in general functions, while the fixed point method only
# converges sugin f(x) and depending on the starting point.