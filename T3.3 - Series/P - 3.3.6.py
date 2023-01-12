"""
Problem 3.3.6 - Continued Fraction.
Approximating exp with continued fraction.
Diego
"""
import numpy as np

def eContinuedFraction(n):
    """Returns the continued fraction value of e for a given number of steps"""
    a = n+1
    for k in range(n, 0, -1): 
        a = k + k/a
    return 2 + 1/a

def eToPrecision(eps=None, maxiter=1e4):
    """Generates e aproximated values until a given presition."""

    if eps == None: #Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    i = 1
    while True:
        
        e = eContinuedFraction(i)

        if abs(e-eContinuedFraction(i-1)) < eps:
            print(f"e = {e} ({i} iterations)")
            break
        elif i > maxiter:
            print("Max interations surpassed. Does not converge.")
            break
        i += 1

eToPrecision(1e-6)
print(np.e)