### Exercise 6.3 (Tiro parabólico)

import numpy as np
import matplotlib.pyplot as plt


def eulerCX(x,f,dt,mode):
    """Combined Euler method for differential equations."""
    if mode.lower() == "simple": a=1;b=0;d=0;g=0
    elif mode.lower() == "modified": a=0;b=1;d=0.5;g=0.5
    elif mode.lower() == "improved": a=0.5;b=0.5;d=1;g=1

    #Since f does not depend on the time, dealing only with x increments does the job
    xt = x + dt*(a*f() + b*f())
    return xt

def eulerCY(y,t,f,dt,mode):
    """Combined Euler method for differential equations."""
    if mode.lower() == "simple": a=1;b=0;d=0;g=0
    elif mode.lower() == "modified": a=0;b=1;d=0.5;g=0.5
    elif mode.lower() == "improved": a=0.5;b=0.5;d=1;g=1

    #Since f does not depend on the time, dealing only with x increments does the job
    yt = y + dt*(a*f(t) + b*f(t+g*dt))
    return yt

def fx(): return vxo

def fy(t): return vyo - g*t

# Initial conditions
g = 9.81
xo,yo = 0,0
vo = 10
theta = np.radians(45)
vxo,vyo = vo*np.cos(theta), vo*np.sin(theta)
dt = 0.01

time = np.arange(0,2,dt)

# Main integration loop
methods = ["Simple", "Modified","Improved"]
posX = [[xo] for _ in range(3)]
posY = [[yo] for _ in range(3)]
for t in time:
    for i,method in enumerate(methods):
        xi = posX[i][-1]
        yi = posY[i][-1]
        x_next = eulerCX(xi,fx,dt,mode=method)
        y_next = eulerCY(yi,t,fy,dt,mode=method)

        posX[i].append(x_next)
        posY[i].append(y_next)


# Plot Settings
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

colorsX = ["magenta", "purple", "violet"]           
colorsY = ["darkgoldenrod", "saddlebrown", "peru"]  
for x,y,i in zip(posX,posY,range(3)):
    # print(temp)
    ax2.plot(time,x[:-1], colorsX[i])
    ax2.plot(time,y[:-1], colorsY[i])
    ax1.plot(x,y)
    
# Exact solution
xExact = vxo*time + xo
yExact = -0.5*g*time**2 + vyo*time + yo
ax1.plot(xExact,yExact, c="k")

ax1.set_xlabel("x");ax1.set_ylabel("y")
ax2.set_xlabel("time (s)");ax2.set_ylabel("coordinates")
ax2.legend(["x","y"])

# plt.legend(["Simple","Modified","Improved", r"Exact"])
plt.tight_layout()
plt.show()

# In the resulting graphs, one can see, as expected, that the position of y respect to 
# x and t is parabolic since the gravity is considered, while for x, its linear since only depends on t.
# The simple euler mehthod is the one that deviate more from exact.


