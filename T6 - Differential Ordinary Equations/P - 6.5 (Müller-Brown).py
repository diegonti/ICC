### Exercise 6.5 - IRC

""" 
First some modules are imported. One of the most importan ones is `SymPy` 
which will allow us to anallitically calculate the gradient and hessian of the MB energy function. 
SymPy works with "symbols" which are equivalent to the variablabes and paramaters used in mathematics. 
This is a version using a unique Python script, but its much interactive and visual to see the procedure and 
expressions in the attached notebook (XXX.ipynb).
"""

# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sm


# Importing some useful SymPy simbols and functions
from sympy.abc import x,y,i,s,A,a,b,c
from sympy import Indexed
xo,yo = sm.symbols("xo,yo")
E = sm.symbols('E', cls=sm.Function)

""" 
The energy function of the MB surface is given by the following expression:
sum from 0 to 3 [ A_i \exp{(a_i(x-x^0_i)^2 + b_i(x-x^0_i)(y-y^0_i) + c_i(y-y^0_i)^2)} ]
Where the A_i, a_i, b_i, c_i, x^0_i, y^0_i are given in the following arrays: 
"""

# Muller-Brown parameters
Ai = np.array((-200,-100,-170,15))
ai = np.array((-1,-1,-6.5,0.7))
bi = np.array((0,0,11,0.6))
ci = np.array((-10,-10,-6.5,0.7))
xoi = np.array((1,0,-0.5,-1))
yoi = np.array((0,0.5,1.5,1))


# The energy expression can be created with the SymPy module
s = Indexed(A,i)* sm.exp(Indexed(a,i)*(x-Indexed(xo,i))**2 + Indexed(b,i)*(x-Indexed(xo,i))*(y-Indexed(yo,i)) + Indexed(c,i)*(y-Indexed(yo,i))**2)
E = sm.Sum(s,(i,0,3))

""" 
The objective of this exercise is to find the Intrinsic Reaction Coordinate (IRC) 
that moves from the transition state (TS) to the lowest minimum.
To accomplish that, an initial direction must be given to start the integration. 
The IRC is computed through the gradient, using the ODE: dr/dt = -g(r)
Where r is the position vector with x and y components. And the initial direction will be 
that the gradient at the initial point has to be the eigenvector of the negative eigenvalue of the hessian. 
Thus, the Gradient vector and the Hessian must be calculated. 
"""

""" 
Once we have the expression for the energy as an analytical SymPy function, 
we can easily compute (anallitically) its gradient vector and its hessian matrix. 
"""
# Gradient vector calculation
g = sm.Matrix([E.diff(coord) for coord in (x,y)])

# Hessian Matrix calculation
H = sm.hessian(E, [x, y])

""" 
With SymPy it is very easy to change the symbolic expressions to usable functions, using lambdify().
"""
# Lambdifying funcitons (turning analytic expressions into numpy functions)
fE = sm.lambdify([x,y,a,b,c,A,xo,yo],E)
fg = sm.lambdify([x,y,a,b,c,A,xo,yo],g)
fH = sm.lambdify([x,y,a,b,c,A,xo,yo],H)

def energy(r): return fE(*r,ai,bi,ci,Ai,xoi,yoi)
def grad(r): return fg(*r,ai,bi,ci,Ai,xoi,yoi).flatten()
def hess(r): return fH(*r,ai,bi,ci,Ai,xoi,yoi)


""" 
The initial point of our IRC calculation is the TS,  r = (-0.822,0.624). 
In that point, the gradient and the hessian can be calculated: 
"""
# Initial point (TS)
r0 = np.array([-0.822,0.624])

# Gradient at initial point
go = fg(*r0,ai,bi,ci,Ai,xoi,yoi)
go = go.flatten()
print("\nGradient at TS: ",go)

# Hessian at initial point
Ho = fH(*r0,ai,bi,ci,Ai,xoi,yoi)
print("\nHessian at TS: ")
print(Ho)


# The eigenvalues and eigenvectors of the Hessian at the TS can be computed with the numpy linal.eig() funciton:
eval,evec = np.linalg.eig(Ho)
print("\nEigenvalues:", eval)
print("Eigenvectors:")
print(evec)

""" 
The first eigenvectors, the correspondent to the negative eigenvalue, 
will be the direction for the first step of the IRC calculations.
Now that we have the initial conditions, the Runge-Kutta4 integrator is used 
to integrate the gradient of the position r to find the IRC.
Note that for the initial direction, a factor is multiplied to $dt$ to get the new position. 
That's because numerical calculations in Python are not that accurate and a little initial "push out" 
is needed to find be correctly aligned to the minimum. Otherwhise, the IRC falls to the relative minimum (intermediate) 
"""

# Runge-Kutta4 method
def rungeKutta4(x,dt,f):
    f0 = f(x)
    f1 = f(x + f0*dt/2)
    f2 = f(x + f1*dt/2)
    f3 = f(x + f2*dt)
    xt = x + dt/6*(f0 + 2*f1 + 2*f2 + f3)
    return xt

# Function to integrate (gradient of position vector r)
def fgrad(r): return -grad(r)


## --- IRC calculation --- ##

dt = 0.00001                    # Time step
rdir = evec[0]*200*dt+r0        # Initial direction using negative eigenvector of the Hessian
positions = [r0,rdir]           # Positions list

# Main integration loop
for i in range(10000):
    ri = positions[-1]                  # Takes the last position
    rNext = rungeKutta4(ri,dt,fgrad)    # Integrates the next one with RK4
    positions.append(rNext)             # Saves position to list
positions = np.array(positions)


# Plot

# MB energy function
def E(x,y):
    energy = 0
    for i in range(4):
        e = Ai[i]*np.exp(ai[i]*(x-xoi[i])**2 + bi[i]*(x-xoi[i])*(y-yoi[i]) + ci[i]*(y-yoi[i])**2)
        energy += e
    return energy

# Energy surface calculation
x = np.linspace(-1.5,1.)
y = np.linspace(-0.5,2.)
XX,YY = np.meshgrid(x,y)
ener = E(XX,YY)

# Figure for visualization
fig = plt.figure(figsize=(9.5,4))
ax1 = fig.add_subplot(1,2,1, projection="3d")
ax2 = fig.add_subplot(1,2,2)

# Plotting 3D view (surface) and cotour plot
ax1.plot(positions[:,0],positions[:,1],E(positions[:,0],positions[:,1]),"ko",markersize=1,alpha=True)
ax1.plot_surface(XX,YY,ener,alpha=0.9,cmap="turbo")
ax2.contour(XX,YY,ener, levels=50, cmap="RdGy")
ax2.plot(positions[:,0],positions[:,1], c="k", lw=2)
ax2.plot(*r0,"ko",markersize=8)
ax2.plot(*positions[-1],"ko",markersize=8)
ax2.text(*r0+0.1,"TS", fontsize="large")
ax2.text(*positions[-1]+0.1,"Min", fontsize="large")

ax1.set_xlim(-1.5,1);ax1.set_ylim(-0.5,2);ax1.set_zlim(-150,100)
ax1.set_xlabel("x");ax1.set_ylabel("y");ax1.set_zlabel("E")
ax2.set_xlabel("x");ax2.set_ylabel("y")
ax1.view_init(56, -150)

fig.tight_layout(h_pad=3,w_pad=5)

plt.show()