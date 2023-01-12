"""
Problem 6.4 - Animal Poblation.
Solving ODE (Lotka-Volterra) with RungeKutta4.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt

def rungeKutta4(x,y,dt,f):
    f0 = f(x,y)
    f1 = f(x + f0*dt/2,y)
    f2 = f(x + f1*dt/2,y)
    f3 = f(x + f2*dt,y)
    xt = x + dt/6*(f0 + 2*f1 + 2*f2 + f3)
    return xt

def fx(x,y): return a*x - b*x*y

def fy(y,x): return -g*y + d*x*y

# Initialization parameters
a,b,g,d = 0.04,0.0005,0.3,5e-5
xo, yo = 1000,40
dt = 0.01

time = np.arange(0,1000,dt)

# Main integration loop
Xlist = [xo]
Ylist = [yo]
for t in time:
    xi = Xlist[-1]
    yi = Ylist[-1]
    x_next = rungeKutta4(xi,yi,dt,fx)
    y_next = rungeKutta4(yi,xi,dt,fy)

    Xlist.append(x_next)
    Ylist.append(y_next)

# Plot Settingds
fig = plt.figure(figsize=(7,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

try:
    scaleX = max(Xlist); scaleY = max(Ylist)
    image = plt.imread("rabbit.jpg")
    ax2.imshow(image,extent=(scaleX*0.75,scaleX*0.99,scaleY*0.75,scaleY*0.99),aspect="auto", zorder=-1)
except FileNotFoundError: pass

ax1.plot(time,Xlist[:-1])
ax1.plot(time,Ylist[:-1])
ax2.plot(Xlist,Ylist)

ax1.legend(["Rabbits","Wolves"])
ax1.set_title("Time evolution")
ax2.set_title("Parametric dependence")
ax1.set_xlabel("time (months)");ax1.set_ylabel("animals")
ax2.set_xlabel("rabbits (x)");ax2.set_ylabel("wolves (y)")

plt.tight_layout()
plt.show()

# The results show periodic evolution of the poblations of rabbits and wolves
# When the population of wolves increases the population of rabbits lowers, since they eat the rabbits.
