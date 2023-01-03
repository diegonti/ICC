"""
Problem 3 - EDPs
Time propagation of Schrodinger Equation.
Diego Ontiveros
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def tridiagonal(dim,k_diagonal,k_below,k_above): 
    """Creates tridiagonal matrix with specified values."""
    a = np.ones((1, dim))[0]*k_diagonal
    b = np.ones((1, dim-1))[0]*k_below
    c = np.ones((1, dim-1))[0]*k_above
    m = np.diag(a, 0) + np.diag(b, -1) + np.diag(c, 1)
    return m



def V(x,y):
    D0 = 33714
    D1 = 303435
    xe,ye = 1,0
    a = 2

    return D0*(1-np.exp(-a*(y/2 - x - xe)))**2 + D0*(1-np.exp(-a*(y/2 + x - xe)))**2 + 0.5*D1*(y-ye)**2

def TOperator(H,dt):
    return np.exp(-1j*dt*H/hbar)
    
def wavepacket(x,x_mean,s2,p):
    """x_mean = <x>, s2 = <(dx)^2>, p = <p>/hbar"""
    return np.array(1/(2*pi*s2)**0.25 * np.exp(1*p*(x-x_mean))*np.exp(-(x-x_mean)**2/(4*s2))) #############

def phi0(x,mu,w):
    hbar=1
    return (mu*w/(pi*hbar))**0.25 * np.exp(-mu*w*x**2/(2*hbar))

# CONSTANTS and REDUCED UNITS
pi = np.pi
hbar = 1.054571817e-34
uma = 1.6605402e-27
me = 9.1093837015e-31
a0 = 5.29177210903e-11
e = 1.602176634e-19
Eh = hbar**2/(me*a0**2)

# Units and Parameters
x_mean = 5.5
s2 = 0.04
p = 5.5
mu = 2/3 * 1836
w = 7.535e14 * hbar/Eh




# Space Conditions

x0 = -10
L = 20                     # Length (au)
J = 100                   # gridpoints
dx = (L)/(J)                # subintervals of length dx
x_grid = np.arange(x0,(x0+L)+dx,dx)

# Time Conditions
dt = 1.9791e-16 * Eh/hbar       # Time step (au)
N = 1000                        # Time points
t = N*dt                        # Total time
t_grid = np.arange(0,t+dt,dt)


# Animation parameters
animation_frames = 100
animation_name = "Schrodinger1.gif"

# A matrix
a = dt/(4*mu*dx**2)           # Parameter a 
A = tridiagonal(len(x_grid),2j+2*a,-a,-a)
B = tridiagonal(len(x_grid),2j-2*a,+a,+a)


# Concentration arrays initialization cA = conc[0] and cB = conc[1]
WF = np.zeros(shape=len(x_grid)) 
phi = phi0(x_grid,mu,w)
GWP = wavepacket(x_grid,x_mean,s2,p)
WF = phi+GWP

WF_init = WF.copy()


# Integration loop
WFt = []
print("\nStarting integration...")
for i,ti in enumerate(t_grid):

    # Periodic conditions
    # WF[-1] = WF[0]

    # Saving frames to animate later
    if i%int(N/animation_frames) == 0 : 
        WFt.append(WF.copy())

    # Updating concentrations
    # C = np.zeros(J+1,dtype=complex)
    # C[0] = a*WF[0]
    # C[-1] = a*WF[-1]


    WF = inv(A)@B@WF  #+ dt*inv(A)@(V(x_grid,0)*WF) #+inv(A)@C


WFt = np.array(WFt)


# Creating GIF animation of the evolution of concentrations
print("\nStarting animation...")
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

def Animation(frame):
    """Function that creates a frame for the GIF."""
    ax1.clear();ax2.clear()

    # Initial Wavefunction
    WF0_frame, = ax1.plot(x_grid,WF_init.real,c="blue",alpha=0.3,label="$Re(psi(t_0))$")

    # Wavefunction evolution
    WFt_frame, = ax1.plot(x_grid,WFt[frame].real,c="red",label="$\Im(psi(t))$")


    # Walls
    wall1 = ax1.axvline(x0+L,ymin=0,c="k",alpha=0.5)
    wall2 = ax1.axvline(x0,ymin=0,c="k",alpha=0.5)

    ax1.set_xlabel("x");ax1.set_ylabel("$\psi$")
    ax2.set_xlabel("x");ax2.set_ylabel("$\psi$")
    ax1.set_ylim(ymin=-1)

    ax1.legend(loc="upper right")

    return WF0_frame,WFt_frame

animation = FuncAnimation(fig,Animation,frames=animation_frames,interval=20,blit=False,repeat=True)
animation.save(animation_name,dpi=120,writer=PillowWriter(fps=25))
fig.tight_layout()
plt.show()

