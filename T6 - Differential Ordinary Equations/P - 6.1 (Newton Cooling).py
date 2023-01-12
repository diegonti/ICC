"""
Problem 6.1 - Newton Cooling.
Solving ODE (Newton's cooling equations) with combined Euler method.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt

def eulerC(x,f,dt,mode):
    """Combined Euler method for differential equations."""
    if mode.lower() == "simple": a=1;b=0;d=0;g=0
    elif mode.lower() == "modified": a=0;b=1;d=0.5;g=0.5
    elif mode.lower() == "improved": a=0.5;b=0.5;d=1;g=1

    #Since f does not depend on the time, dealing only with x increments does the job
    xt = x + dt*(a*f(x) + b*f(x+f(x)*d*dt))
    return xt

def f(T): return K*(Tf-T)

K = 0.05
Tf = 298.15
To = 500
dt = 0.01

time = np.arange(1,200,dt)

methods = ["Simple", "Modified","Improved"]
temperatures = [[To] for _ in range(3)]
for t in time:
    
    for i,method in enumerate(methods):
        Ti = temperatures[i][-1]
        T_next = eulerC(Ti,f,dt,mode=method)
        temperatures[i].append(T_next)

for temp in temperatures:
    # print(temp)
    plt.plot(time,temp[:-1])

#T = Tf - + (Tf-To)exp
T = Tf + (To-Tf)*np.exp(-K*time)
plt.plot(time,T)

plt.legend(["Simple","Modified","Improved", r"Exact"])
plt.show()

# As expected, the temperature will increase (or decrease) exponentially 
# unitl arrive to the ambien temperature.
# The different methods present very similar results to the exact resolution.


