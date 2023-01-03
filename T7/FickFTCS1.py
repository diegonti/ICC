"""
Problem 1.1 - EDPs
2nd Fick law 1D FTCS. Constant flow to fill a segment.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter





def integrateFick(D,L,c0,dx,t_range = (0,10), pbc = False, animation=True,animation_frames=100,animation_name="fick.gif"):

    t0,tf = t_range                 # Time range
    dt = 0.5 * dx**2/(2*D)          # time increment
    time = np.arange(t0,tf+dt,dt)   # time grid
    space = np.arange(0,L+dx,dx)    # space grid
    t_points = len(time)            # number of time frames
    x_points = len(space)
    
    assert t_points > animation_frames, f"Number of frames for animations should be lower than the time points ({t_points})"

    print("Number of time points calculated:",t_points)
    print("Number of frames used for the GIF:",animation_frames)

    # Intializing concentration arrays
    c_old = np.zeros(len(space))
    c_new = np.zeros(len(space))

    
    # Initial conditions
    c_old[0] = c0
    c_new[0] = c0

    # Integration loop
    print("Starting Integration...")
    i_frame = 0
    c_init = c_old.copy()
    ct = [0 for _ in range(animation_frames+1)]
    for i,ti in enumerate(time):

        # Boundary Conditions

        if pbc: # Periodic 
            c_new[0] = c_new[0] + D*dt/dx**2 * (c_new[-1] + c_new[1] - 2*c_new[0])
            c_new[-1] = c_new[-1] + D*dt/dx**2 * (c_new[0] + c_new[-2] - 2*c_new[-1])
        else:   # Walls
            c_new[0] = c0
            c_new[-1] = c_new[-1] + D*dt/dx**2 * (0 + c_new[-2] - 2*c_new[-1])

        for j, xi in enumerate(space,start=1):
            try:
                c_new[j] = c_old[j] + D*dt/dx**2 * (c_old[j+1] + c_old[j-1] - 2*c_old[j])
                c_old[j] = c_new[j]
            except IndexError: pass
        c_old = c_new

        if i%int(t_points/animation_frames) == 0 : 
            ct[i_frame] = np.array(c_new)
            i_frame +=1
    ct = np.array(ct,dtype=object)
   

    if animation:
        print("Starting animation...")
        # Creating GIF animation of the evolution of concentrations
        fig,ax = plt.subplots()

        def Animation(frame):
            """Function that creates a frame for the GIF."""
            ax.clear()
            c0_frame, = ax.plot(space,c_init,c="orange",alpha=0.5,label="initial")
            ct_frame, = ax.plot(space,ct[frame],c="red",label="over time")
            wall1 = ax.axvline(L,ymin=0,c="k",alpha=0.5)
            wall2 = ax.axvline(0,ymin=0,c="k",alpha=0.5)
            ax.set_xlabel("x");ax.set_ylabel("c")
            ax.legend()
            return c0_frame,ct_frame

        animation = FuncAnimation(fig,Animation,frames=animation_frames,interval=20,blit=True,repeat=True)
        animation.save(animation_name,dpi=120,writer=PillowWriter(fps=25))
        plt.show()
        
    else:
        # Plotting Initial vs. Final concentration profiles
        fig,ax2 = plt.subplots()

        ax2.plot(space,c_init,c="orange",alpha=0.5,label="initial")
        ax2.plot(space,c_new,c="red",label="over time")
        ax2.grid(alpha=0.25)
        ax2.axvline(L,ymin=0,c="k",alpha=0.5)
        ax2.axvline(0,ymin=0,c="k",alpha=0.5)

        ax2.set_xlabel("x");ax2.set_ylabel("c")
        ax2.legend()

    return fig

# Simulation parameters
fig = integrateFick(
    D = 0.01,
    c0 = 1,
    L = 1,
    dx = 0.01,
    t_range = (0,10),
    pbc=False,
    animation=True,animation_frames=100,animation_name="fick1.gif"
    )
plt.show()

# The flux begins to fill the box.