"""
Problem 3.3.3 (2) - Continued fraction.
Finding roots of quadratic equation with continued fractions.
Diego Ontiveros
"""
def continuedFraction(a,b,c, eps=None,maxiter=1e3):
    """Returns roots of quadratic equation by continued fraction."""
    
    if eps == None: # Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    # Loop for root1
    i = 0
    root1,root2 = -1,1
    while True:
        root1 = -b - c/root1
        next = -b - c/root1

        if abs(root1 - next) < eps:
            print(f"Root 1 : {root1}")
            break
        elif i > maxiter:
            print("Max interations surpassed. Does not converge.")
            break
        i += 1
    
    # Loop for root2
    i = 0
    while True:
        root2 = -c/(b+root2)
        next = -c/(b+root2)

        if abs(root2 - next) < eps:
            print(f"Root 2 : {root2}")
            break
        elif i > maxiter:
            print("Max interations surpassed. Does not converge.")
            break
        i += 1

    return root1,root2

continuedFraction(1,4,-1)
