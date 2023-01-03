### Exercise 4.1.1

#Getting machine epsilon
macheps = 1
while macheps+1.0>1.0: macheps = macheps*0.5

#Differentiation functions
def backward(f,x, eps=None):
    """Backwards differentiation of function f"""
    if eps == None: eps = macheps #Get macheps if eps not specified
    h = eps

    f_prime = (f(x)-f(x-h))/h

    return f_prime

def forward(f,x, eps=None):
    """Forward differentiation of function f"""
    if eps == None: eps = macheps #Get macheps if eps not specified
    h = eps

    f_prime = (f(x+h)-f(x))/h
    
    return f_prime

def central(f,x, eps=None):
    """Central differentiation of function f"""
    if eps == None: eps = macheps #Get macheps if eps not specified
    h = eps

    f_prime = (f(x+h)-f(x-h))/(2*h)
    
    return f_prime

import numpy as np
import matplotlib.pyplot as plt

def f(x): return np.sin(x)
def df(x): return np.cos(x)

eps = 1e-1

x = np.arange(0,2*np.pi, eps)
fx,dfx = f(x), df(x)
f_primeB = backward(f,x,eps)
f_primeF = forward(f,x,eps)
f_primeC = central(f,x,eps)

#Plot Settings
plt.plot(x,df(x), label = "cos(x)")
plt.plot(x,f_primeB, label = "backward")
plt.plot(x,f_primeF, label = "forward")
plt.plot(x,f_primeC, label = "central")
plt.legend(); plt.xlabel("x"), plt.ylabel("y")
plt.axhline(0, color="black", lw=0.75)
plt.xlim(x[0],x[-1])
plt.show()

#Looking at the produced plot, one can see that the cental differentiation is the one that
#ajusts best to the real derivate, while backwards and forward differentiation are shifted 
#forward or back, respectively.


##Generating file with the data of the derivatives
# dig=8 #digits to export
# with open("diff.txt","w") as outfile:
#     outfile.write(f"x   f(x)    backward(x)    forward(x)  central(x)\n")
#     for i,p in enumerate(x):
#         outfile.write(
#             f"{p:.{dig}f}   {fx[i]:.{dig}f} {f_primeB[i]:.{dig}f}   {f_primeF[i]:.{dig}f}   {f_primeC[i]:.{dig}f}\n")
