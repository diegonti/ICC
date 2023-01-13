"""
Problem 7.2 - Diffusion-Reaction System
Solving Diffusion-Reaction System with implicit method.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def tridiagonal(dim,k_diagonal,k_below,k_above): 
    """Creates tridiagonal matrix with specified values."""
    a = np.ones((1, dim))[0]*k_diagonal
    b = np.ones((1, dim-1))[0]*k_below
    c = np.ones((1, dim-1))[0]*k_above
    m = np.diag(a, 0) + np.diag(b, -1) + np.diag(c, 1)
    return m

def initC(conc:np.ndarray,cA,cB):
    """Initializes Concentrations"""
    _,x_points = conc.shape
    s = np.zeros(conc.shape)

    # Initializing concentrations of A (0)
    s[0][1:int(x_points/3)] += cA
    s[0][int(2*x_points/3):-1] += cA

    # Initializing concentrations of B (1)
    s[1][int(x_points/3):int(2*x_points/3)] += cB

    return s

    
# Space Conditions
L = 1                     # Length
J = 100                   # gridpoints
dx = L/(J)                # subintervals of length dx
x_grid = np.arange(0,L+dx,dx)

# Time Conditions
t = 2000                  # Total time
N = 1000                  # Time points
dt = t/(N-1)              # Subintervals
t_grid = np.arange(0,t+dt,dt)

k = 1e-3                  # velocity constant

# Animation parameters
animation_frames = 100
animation_name = "DiffusionReaction.gif"

# A matrix
D = 0.0001               # Diffusion coefitient
a = D*dt/dx**2           # Parameter a 
A = tridiagonal(J+1,1+2*a,-a,-a)
# Settin diagonal extremes to 1 and next element to 0
A[0,0],A[-1,-1] = 1,1
A[0,1],A[-1,-2] = 0,0

print("\nA Matrix (Update diffusion matrix)")
print(A)

# Concentration arrays initialization cA = conc[0] and cB = conc[1]
conc = np.zeros(shape=(2,J+1))  
conc = initC(conc,1,2)
c_init = conc.copy()

# Integration loop
ct = [[],[]]
alpha = []
print("\nStarting integration...")
for i,ti in enumerate(t_grid):

    # Periodic conditions
    conc[0][0] = conc[0][-1] = conc[0][-2]
    conc[1][0] = conc[1][-1] = conc[1][-2]

    # Saving frames to animate later
    if i%int(N/animation_frames) == 0 : 
        ct[0].append(conc[0].copy())
        ct[1].append(conc[1].copy())

        ai = -np.log(conc[0].mean()) / np.log(ti)
        alpha.append(ai)

    # Updating concentrations
    conc[0] = np.linalg.inv(A)@conc[0] - dt*k*conc[0]*conc[1]
    conc[1] = np.linalg.inv(A)@conc[1] - dt*k*conc[1]*conc[1]

ct = np.array(ct,dtype=object)
alpha = np.array(alpha)


# Creating GIF animation of the evolution of concentrations
print("\nStarting animation...")
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

def Animation(frame):
    """Function that creates a frame for the GIF."""
    ax1.clear();ax2.clear()
    # Initial concentrations
    cA0_frame, = ax1.plot(x_grid,c_init[0],c="red",alpha=0.3,label="$c_A^0(t)$")
    cB0_frame, = ax1.plot(x_grid,c_init[1],c="blue",alpha=0.3,label="$c_A^0(t)$")

    # Concentration evolution
    cAt_frame, = ax1.plot(x_grid,ct[0][frame],c="red",label="$c_A(t)$")
    cBt_frame, = ax1.plot(x_grid,ct[1][frame],c="blue",label="$c_B(t)$")

    # Alpha evolution
    alpha_time = t_grid[::int(N/animation_frames)]
    a_frame, = ax2.plot(alpha_time[1:],alpha[1:],c="green")
    step_frame = ax2.axvline(t_grid[int(frame*N/animation_frames)], c="k",alpha=0.5)

    # Walls
    wall1 = ax1.axvline(L,ymin=0,c="k",alpha=0.5)
    wall2 = ax1.axvline(0,ymin=0,c="k",alpha=0.5)

    ax1.set_xlabel("x");ax1.set_ylabel("c")
    ax2.set_xlabel("t");ax2.set_ylabel("$\\alpha$")
    ax2.set_xlim(alpha_time[0],alpha_time[-1])

    ax1.legend(loc="upper right")

    return cAt_frame,cBt_frame, step_frame

animation = FuncAnimation(fig,Animation,frames=animation_frames,interval=20,blit=True,repeat=True)
animation.save(animation_name,dpi=120,writer=PillowWriter(fps=25))
fig.tight_layout()
plt.show()

# With the GIF representation, you can very visually see how both species
# diffuse between each other at the same time the reaction goes and makes both concentrations
# decrease. Playing with the diffusion constant (D) and reaction velocity constant (k)
# one can see which one is controlling the system.

# In the default case, the reaction first is mainly controled by difusion, but as the time progresses,
# both reactants are mixed toghether and the reaction regim begins to take control.
