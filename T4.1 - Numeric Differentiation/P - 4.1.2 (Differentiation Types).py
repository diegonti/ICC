"""
Problem 4.1.2 - Differentiation Types.
Backward, Forward and Central Numerical differentiation of a function.
Diego Ontiveros
"""
#Getting machine epsilon
macheps = 1
while macheps+1.0>1.0: macheps = macheps*0.5

#Differentiation functions
def backward(f,x,h): return (f(x)-f(x-h))/h

def forward(f,x, h): return (f(x+h)-f(x))/h
    
def central(f,x, h): return (f(x+h)-f(x-h))/(2*h)
    

import numpy as np
import matplotlib.pyplot as plt

def f(x): return np.sin(x)
def df(x): return np.cos(x)

eps = macheps
x = 1

#Building h array in log scale
h = np.array([])
n=1
while True:
    step = 1/2**n
    h = np.append(h,step)
    if step < 0.01*eps: break
    n+=1

#Functions and derivatives
fx,dfx = f(x), df(x)
f_primeB = backward(f,x,h)
f_primeF = forward(f,x,h)
f_primeC = central(f,x,h)

#Absolute errors
eabsB = abs(dfx-f_primeB)
eabsF = abs(dfx-f_primeF)
eabsC = abs(dfx-f_primeC)

#Plot Settings
plt.rcParams["figure.autolayout"] = True
fig,ax = plt.subplots(3,1, figsize = (5,6))

ax[0].loglog(h,eabsB, label = "backward")
ax[1].loglog(h,eabsF, label = "forward")
ax[2].loglog(h,eabsC, label = "central")

for i in range(3):
    # ax[i].set_yscale("log");ax[i].set_xscale("linear")
    ax[i].loglog(h,h, label = r"$h$", ls="dotted")
    ax[i].loglog(h,h**2, label = r"$h^2$",ls = "dotted")

    ax[i].set_ylim(ymin = 1e-18)
    ax[i].set_ylabel(r"$E_{abs}$")
    ax[i].legend(fontsize = "small")
    # ax[i].grid(True, which="both")

ax[0].set_title("Absolute Errors for numerical differentiation of sin(1)", fontsize="medium")
plt.xlabel("h")
plt.show()

# All differentiations present a valley-like (in log scale) of the absolute error,
# with the minimum at around h=1e-8 for Backwards and Forwardd and h=1e-6 for Central.
# Morover, from that minumum, the absolute error for Backwards and Forward differetiation seems to grow linearly with h,
# while for central it grows with h**2.