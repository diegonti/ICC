"""
Problem 3.3.1 - Infinite sum.
Approximating exp with infinite sum.
Diego Ontiveros
"""
import numpy as np

def factorial(x):
    """Returns the factorial of a number (x!)."""
    result = 1
    for i in range(1,x+1): result *= i
    return result

def exponential():
    """Calculates number e with its series representation."""

    # Gets the machine epsilon
    eps = 1
    while eps+1.0>1.0: eps = eps*0.5

    # Loop of the infinite sum
    result, i = 0, 0
    while True:
        step = 1/factorial(i)
        result += step 
        if step <= eps:
            Erel = abs(result-np.e)/np.e
            print(f"e = {result}, iterations: {i}, Erel = {Erel}")
            return result
            break
        i += 1

exponential()
