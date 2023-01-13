"""
Problem 3.3.6 - Continued Fraction.
Approximating error function with continued fraction and infinite sum.
Diego Ontiveros
"""
import numpy as np

def factorial(x):
    """Returns the factorial of a number (x!)."""
    result = 1
    for i in range(1,x+1): result *= i
    return result

def erfc_series(x, eps=None, maxiter=1e4):
    """Generates e aproximated values until a given presition."""

    if eps == None: #Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    k,suma = 0,0
    while True:

        mem = suma
        suma += 2/(np.pi**0.5)*((-1)**k * x**(2*k+1) /(factorial(k)*(2*k+1)))
        e = 1 - suma
        eabs = abs(e-(1-mem))

        if eabs < eps:
            print(f"e = {e} ({k} iterations)")
            break
        elif k > maxiter:
            print("Max interations surpassed. Does not converge.")
            break
        k += 1

def erfc_continuedF(x):
    """exp(-x**2)/pi**0.5 * (1/(x+n/2))"""
    pass

print("Using Infinite series...")
erfc_series(1,1e-8)
print("Using Continued fraction...")
erfc_continuedF(1,1e-6)
