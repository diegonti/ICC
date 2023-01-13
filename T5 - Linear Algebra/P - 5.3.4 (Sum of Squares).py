"""
Problem 5.3.4 - Optimization. Sum of Squares
Fitting function points using Gauss-Newton to minimize residuals.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt 
from time import time
time_0 = time()


def eulerC(X,f,t_range,mode="improved"):
    """Combined Euler method for differential equations.
     Integrates function f over a time range t_range"""

    # Choosing integration mode for Euler
    if mode.lower() == "simple": a=1;b=0;d=0;g=0
    elif mode.lower() == "modified": a=0;b=1;d=0.5;g=0.5
    elif mode.lower() == "improved": a=0.5;b=0.5;d=1;g=1

    # Integrates function f over a time range t_range
    y = []
    dt = t_range[1] - t_range[0]
    for i,_ in enumerate(t_range):
        #Since f does not depend on the time, dealing only with x increments does the job
        if i == 0: yt = X[2]
        else: yt = y0 + dt*(a*f(y0,X) + b*f(y0+f(y0,X)*d*dt,X))
        
        y.append(yt)
        y0 = yt

    return y


def residual(x0,t_range,t,y):
    """Calculates residuals."""

    f = eulerC(x0,y_prime,t_range)
    dt = t_range[1] - t_range[0]

    DY = []
    for yi,ti in zip(y,t):
        # ti/dt chooses the right index that corresponds to the integrated function
        dy = yi - f[int(ti/dt)]
        DY.append(dy)
    DY=np.array(DY)

    return DY


def Jacobian(X,t,t_range,step=1e-6):
    """Computes Jacobian"""

    dt = t_range[1] - t_range[0]

    X0 = X
    X1 = np.array([X[0]+step,X[1],X[2]])
    X2 = np.array([X[0],X[1]+step,X[2]])

    z0 = eulerC(X0, y_prime,t_range)
    z1 = eulerC(X1, y_prime,t_range)
    z2 = eulerC(X2, y_prime,t_range)

    GRAD_X1 = []; GRAD_X2 = []
    for ti in t:
        grad_x1 = (z1[int(ti/dt)]-z0[int(ti/dt)])/step
        grad_x2 = (z2[int(ti/dt)]-z0[int(ti/dt)])/step
        GRAD_X1.append(grad_x1); GRAD_X2.append(grad_x2)
    return np.column_stack([np.array(GRAD_X1),np.array(GRAD_X2)])


def GaussNewton(t,y,X,tol=1e-6,max_iter=100):
    """Uses GaussNewton to optimize."""

    old = new = np.array(X)
    for i in range(max_iter):
        old = new

        J = Jacobian(old,t,t_range)
        DY = residual(old,t_range,t,y)

        update = np.linalg.inv(J.T@J) @ J.T @ DY

        new[0]=old[0]+update[0]; new[1]=old[1]+update[1]

        if np.linalg.norm(old-new) < tol:
            return new


def y_prime(y, X): return -X[0]*y/(X[1]+y)


############## MAIN PROGRAM ################

# Initial data
t = np.array([0, 23.6, 49.1, 74.5, 80.0, 100.0, 125.5, 147.3])
y = np.array([24.44, 19.44, 15.56, 10.56, 9.07, 6.85, 4.07, 1.67])

inital_guess = np.array([0.22, 3.27, 24.44])

# Optimization
t0,tf = 0,150                   
h = 0.001                       # Timestep
t_range=np.arange(t0, tf, h)    # Time range 

coeff = GaussNewton(t,y,inital_guess,tol=1e-8)
print(f"Optimized parameters: X = ",coeff)

# Plotting Settings
plt.plot(t_range, eulerC(coeff,y_prime,t_range),"k:",label="fitting")
plt.scatter(t,y,c="r",label="data points")


plt.xlabel("t");plt.ylabel("y")
plt.legend()

time_f = time()
print(f"Process finished in {time_f-time_0:.2f}s.")
plt.show()