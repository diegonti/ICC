### Exercise 3.3.3 (solve with fixed-point method)


def fixedPoint(f,n,a,start,eps=None,maxiter=1e5):
    
    if eps == None: #Get macheps if eps not specified
        eps = 1
        while eps+1.0>1.0: eps = eps*0.5

    #Method
    x,i=start,0
    while True:
        
        step = f(x,n,a)
        eabs = abs(x-step)
        x=step
        # print(x)
        
        if eabs < eps:
            print(f"Root with Method1: {x}. Iterations: {i}, Eabs={eabs}")
            break
        elif i > maxiter:
            print("Max interations surpassed. Does not converge.")
            print(f"The number until now is: {x}")
            break
        i += 1
        
def f1(x,n,a): return a/(x**(n-1))
def f2(x,n,a): return ((n-1)*x + a/(x**(n-1)))/n

fixedPoint(f1, 1.3,1.1,1, eps=1e-6)
fixedPoint(f2, 1.3,1.1,1, eps=1e-6)
#Method 1 diverges when n>=2, and does not perform well with very low epsilons.
#Method 2 converges much faster and in many more cases.
