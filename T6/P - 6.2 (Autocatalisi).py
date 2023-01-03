### Exercise 6.2 (Reacciones autocatalíticas)

import numpy as np
import matplotlib.pyplot as plt

def eulerC(y,x,f,dt,mode):
    """Combined Euler method for differential equations."""
    if mode.lower() == "simple": a=1;b=0;d=0;g=0
    elif mode.lower() == "modified": a=0;b=1;d=0.5;g=0.5
    elif mode.lower() == "improved": a=0.5;b=0.5;d=1;g=1

    #Since an f step does not depend  explicitly on the time, dealing only with x increments does the job
    yt = y + dt*(a*f(y,x) + b*f(y+f(y,x)*d*dt,x))
    return yt

def fx(x,y): return -2*k1*x-2*k2*x*y

def fy(y,x): return 2*k1*x+2*k2*x*y

# Initial conditions
k1 = 1e-4       # s-1
k2 = 8          # M-1s-1
xo = 1e-3       # [MnO4-]o
yo = 0          # [Mo2+]o
dt = 1          
time = np.arange(0,750,dt)  # study time-range

# Main integration loop
methods = ["Simple", "Modified","Improved"]
concentrationsX = [[xo] for _ in range(3)]
concentrationsY = [[yo] for _ in range(3)]
for t in time:
    for i,method in enumerate(methods):

        # Concentrations will be the last ones saved
        xi = concentrationsX[i][-1]
        yi = concentrationsY[i][-1]

        # Calculating the next concentrations with euler
        x_next = eulerC(xi,yi,fx,dt,mode=method)
        y_next = eulerC(yi,xi,fy,dt,mode=method)
        concentrationsX[i].append(x_next)
        concentrationsY[i].append(y_next)

# Plot and plot settings
colorsX = ["purple", "magenta", "violet"]           # MnO4- characteristic colours
colorsY = ["saddlebrown", "darkgoldenrod", "peru"]  # Mn2+ characteristic colours
for x,y,i in zip(concentrationsX,concentrationsY,range(3)):
    x,y = np.array(x),np.array(y)
    plt.plot(time,x[:-1]*1000, color=colorsX[i])
    plt.plot(time,y[:-1]*1000, color=colorsY[i])

plt.xlabel("time (s)"); plt.ylabel("Concentrations (mM)")
plt.legend([r"MnO$_4^-$", r"Mn$^{2+}$"])

plt.tight_layout()
plt.show()

# The results show the tipycall sigmoid curve that an autocatalitic reaction presents.
# The reactions starts slowly because low concentrations of Mn2+ are available, 
# but with increasing the Mn2+, since it helps to increase the velocity of reaction, 
# makes the reaction go even faster, so a quick growth is shown for the increase of product and decrease of reactant,
# until the concentration of reactant lowers and the rate slows down.

# All methods perform well and give very similar results, but the simple euler method differs a little bit from the others.




