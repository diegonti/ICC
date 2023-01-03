"""
Montecarlo Script to compute the most stable configuration of two water molecules.
It uses a first sampling of the systems box to find stable points, and from
there, starts the metropolis algorithm to find the energy minimum.

With the default parameters (sampling of 1e5 steps and montecarlo of 1e5 steps)
the program runs in ~30s and returns energies around -6.5 kcal/mol (pretty close to expected)
Using samplig of 1e5 and 1e6 for metropolis runs in ~2mins.

For more info, see also the testingMC.ipynb notebook, were different tests were performed,
(scaling and velocity tests, functions tests, visualization, etc).

Diego Ontiveros Cruz -- 15/11/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
to = time()

# Energies parameters (VdW A and B, and coulomb charges)
# In matrix form since it will be easier and faster to compute later
# Each A,B and Q contain all the combinations ij of Aij,Bij and qij.
A = np.array([[581935.563838,328.317371,328.317371],
            [328.317371,9.715859e-6,9.715859e-6],
            [328.317371,9.715859e-6,9.715859e-6]])
B = np.array([[594.825035,10.478040,10.478040],
            [10.478040,0.001337,0.001337],
            [10.478040,0.001337,0.001337]])
q = np.array([-0.834,+0.417,+0.417])
Q = np.dot(q[:,None],q[:,None].T)


def acceptE(dE):
    """
    Accepts or regects a Monte Carlo step.

    Parameters
    ----------
    `delta_e` : difference in energy from one step to another

    `kT` : factor to consider 

    Returns
    ----------
    `True` : if step is accepted 
    
    `False` : if step is not accepted       
    """
    if dE < 0: 
        return True
    else:
        rand = np.random.random()
        if np.exp(-beta*dE) > rand:
            return True
        else:
            return False
            

class Water():
    """
    A tip3p Water molecule class.
    """
    def __init__(self):
        self.rO = np.array([0.,0.,0.])
        self.rH1 = np.array([0.75669,0.58589,0.])
        self.rH2 = np.array([-0.75669,0.58589,0.])
        self.coord = np.array([self.rO,self.rH1,self.rH2])
        

    def translate(self,Tx,Ty,Tz):
        """Translates the molecule object to a specified point for each main coordinate.
        
        Parameters
        ----------
        `Tx,Ty,Tz` : translation step on the x,y,z coordinates, respectively.
        
        """
        self.coord = self.coord.T   # Transpose returns arrays by coordinate (X,Y,X) instead of (O,H1,H2)
        for i,T in enumerate([Tx,Ty,Tz]):
            self.coord[i] += T
        self.coord = self.coord.T


    def rotate(self,Ax,Ay,Az, atype="deg"):
        """Rotates the molecule object a specified angle for each main axis.

        Parameters
        ----------
        `Ax,Ay,Az` : rotation along the x,y,z axis, respectively.

        `atype` : angle type. "deg" for the input in degrees (default), "rad" for the input in radians.
        
        """
        # Angle type 
        atype = atype.lower()
        if atype == "deg" or atype == "d": Ax,Ay,Az = np.radians(Ax),np.radians(Ay),np.radians(Az)
        elif atype == "rad" or atype == "r": pass
        else: raise TypeError("Angle type not detected. Choose deg for degrees or rad for radians.")

        # Rotation matrices
        Rx = np.array([[1,0,0],[0,np.cos(Ax),-np.sin(Ax)],[0,np.sin(Ax),np.cos(Ax)]])
        Ry = np.array([[np.cos(Ay),0,np.sin(Ay)],[0,1,0],[-np.sin(Ay),0,np.cos(Ay)]])
        Rz = np.array([[np.cos(Az),-np.sin(Az),0],[np.sin(Az),np.cos(Az),0],[0,0,1]])

        # Moves to origin #OHH
        temp = [self.coord[0][i] for i in range(3)] # Saves temporal coordinates of Oxygen, to later replace it on the same spot it was
        self.toOrigin()

        # Rotates the molecule the specified angles
        for i in range(3):
            for R in [Rx,Ry,Rz]:
                ctemp = self.coord[i].copy()
                self.coord[i] = np.dot(R,ctemp) 
                #R = np.dot(Rx,Ry,Rz) does not work 100%

        # Returns the molecule to the same place before (but rotated)
        self.translate(*temp) 


    def toOrigin(self):
        """Moves the molecule to the origin of coorinated (centered in the Oxygen)"""
        c = self.coord[0] 
        self.translate(-c[0],-c[1],-c[2])

       
    def getEnergies(self,other):
        """Gets the energy (kcal/mol) of a pair of molecules.
        
        Parameters
        ----------
        `other` : other water molecule object.

        Returns
        ----------
        `E` : total energy, E=Eelec+Evdw 
        
        `Eelec` : Electronic energy.

        `Evwd` : Van der Waals energy.
        """
        # Getting molecule coordinates 
        coord1,coord2 = self.coord,other.coord

        # Combination of distances between all self and other atoms
        diff = coord1-coord2[:,None]                    # coordinates diference (ri-rj) vectors in matrices
        distances = np.sqrt((diff**2).sum(axis=-1)).T   # distances array with each pair of atoms
        
        # Energies calculation. With np.arrays the problem reduces to matrix calculation (faster)
        Evdw = (A/distances**12 - B/distances**6).sum()     
        Eelec = (332.0*Q/distances).sum()
        E = Evdw + Eelec
        self.E = E
        return E,Eelec,Evdw


    def draw(self,ax):
        """Draws water molecule object to a specified axis."""
        ax.scatter(self.coord.T[0],self.coord.T[1],self.coord.T[2], sizes=(100,75,75), c=("red","grey","grey"), alpha=True)


    def reset(self): self.E = 0
    

    def _distances(self):
        """Test function to compute distances and see if is mantained rigid."""
        self.OH1 = np.linalg.norm(self.coord[1]-self.coord[0])
        self.OH2 = np.linalg.norm(self.coord[2]-self.coord[0])
        print(f"Computed: {self.OH1:.3f},{self.OH2:.3f}")
        print("Sould be: ", 0.957)


########################### ----- Main Program ----- #########################      
##############################################################################

# Opening visualization and creating figure axes
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, (3,4))
ax3 = fig.add_subplot(2, 2, 2, projection='3d')

# Simulation parameters
T=300.              # Temperature
kb = 0.00119872041  # Boltzman constant (in kcal/(mol⋅K))
beta = 1./(kb*T)    # Beta factor

# Initialization of fixed water molecule at center
wFixed = Water()
wFixed.draw(ax1)
wFixed.draw(ax3)


######################## ----- Initial MC Sampling ----- #########################   

"""
Initial sampling of the box to get a startig point for metropolis.
This step is done with n-1 moving waters. They only intecract with the fixed one,
can be thought of doing the sampling loop with n-1 different independent waters and
from the results gathering the one that presents the lowest minimum as starting point for metropolis.
"""

# Sampling parameters
n = 2           # Number of waters (1 fixed and n-1 moving)
N = 100000      # Number of iterations
lim = 5         # Box limit

# List with the moving water molecules
waters = [Water() for _ in range(n-1)]

# Lists where energies and coordinates will be saved
energies = [[] for _ in range(n-1)]
coords = [[] for _ in range(n-1)]

# Main Sampling loop with random configurations generated in the box
print("\nStarting initial Sampling ...")
for j in range(N):
    for i,w in enumerate(waters):

        # Random position and angles values
        randT = np.random.uniform(-lim,lim, size=3)     # To random position in a (-lim,lim) box
        randA = np.random.uniform(-180,180,size=3)      # With angles between (-180,180) degrees
        
        # Translation of waters
        w.translate(*randT)

        # Rotation of waters
        w.rotate(*randA,"d")

        # Energy calculations (each water with respect only with the fixed one at the center)
        E,Eelec,Evdw = w.getEnergies(wFixed)

        # Saving energy and coordinates to array
        energies[i].append(E)
        coords[i].append(w.coord.copy())

        # Resets energy and positions
        w.reset()
        w.toOrigin()

# Get minimum Energy and Coordinates from the initial sampling 
energies = np.array(energies)
minEs = np.min(energies,axis=-1)        # Mininum energies from sampling pairs
minEsi = np.argmin(energies,axis=-1)    # Minimum energies indeces
minE = np.min(minEs)                    # Mnimum E from all cases
minEi = np.argmin(minEs)                # Index of min Es in minEs recap list
print("Minimum Energies from sampling (kcal/mol): ", *minEs)

coords = np.array(coords)
minCoords = []
for i in range(n-1):
    # Saving and visualizing the minimum configurations gathered
    wMin = Water() 
    minc = coords[i][minEsi[i]]
    wMin.coord = minc
    minCoords.append(minc)
    wMin.draw(ax1)


######################## ----- Metropolis MC ----- #########################   

"""
Metropolis Monte Carlo to find the minimum configuration from the initial starting point
found in the sampling. Now the molecule moves a given random step and rotates a random angle,
and new configurations are gathered through the Metropolis algorithm.
"""

# Water from the sampled aproximated minimum
wMin = Water()
wMin.coord = minCoords[minEi]

# Lists where energies and coordinates will be saved (starting from lowest state found in sampling)
energies2 = [minE]
elecs, vdws = [],[]
coords2 = [wMin.coord.copy()]

# Metropolis Parameters
N = 100000              # Number of iterations
step = 0.1              # Translation step size
angle = 25              # Angle step size
acceptance = [0,0]      # Acceptance
nPoints = [0]           # List of accepted steps (to plot)
w = wMin

# Main Metropolis MC loop
print("\nStarting Metropolis ...")
for i in range(N):

    # Random steps and rotations
    randT = np.random.uniform(-step,step,size=3)        # Random steps of (-step,step) lenghts
    randA = np.random.uniform(-angle,angle,size=3)      # With rotations between (-angle,angle) degrees
    
    # Translation of water
    w.translate(*randT)

    # Rotation of water
    w.rotate(*randA,"d")

    # Energy calculations (with the fixed water at the origin)
    E,Eelec,Evdw = w.getEnergies(wFixed)
    dE = E - energies2[-1]  # Energy diference

    # Acceptance of the step
    if not acceptE(dE):
        acceptance[1] += 1              # Increase 1 to the not accepted steps
        w.coord = coords2[-1].copy()    # Return to the last configuration (exclude the current one)
        continue                        # continue to next step
    else: 
        acceptance[0] += 1              # Increase 1 to the accepted steps

        # Saving energy and coordinates to array
        energies2.append(E)
        elecs.append(Eelec)
        vdws.append(Evdw)
        coords2.append(w.coord.copy())
        nPoints.append(i+1)

    # Resets energy
    w.reset()

# Energies Outputs
Emin = np.min(energies2)
Emini = np.argmin(energies2)
coordMin = coords2[Emini]
print(f"Accepted: {acceptance[0]}, Not Accepted: {acceptance[1]}. N steps: {sum(acceptance)}")
print(f"Acceptance:                              {100*acceptance[0]/N:.2f}%")
print(f"Energy at final iteration (kcal/mol):    {energies2[-1]:.3f}")
print(f"Minimum energy found (kcal/mol):         {Emin:.3f}")
print(f"Energy contributions (kcal/mol):         E = {Emin:.3f}   Eelec = {elecs[Emini-1]:.3f}   Evdw = {vdws[Emini-1]:.3f}")

#Energy Plot settings
ax2.plot(nPoints,energies2, lw=1)
ax2.set_xlabel("N");ax2.set_ylabel("Energy (kcal/mol)")
ax2.set_title("Energy variation in MonteCarlo Metropolis")

# Min Configuration Visualization
wMin = Water()
wMin.coord = coordMin
wMin.draw(ax3)


# Plot Settings
for ax in [ax1,ax3]:
    ax.quiver(-lim,0,0, 2*lim,0,0, color='k', lw=1, arrow_length_ratio=0.05)   # x-axis
    ax.quiver(0,-lim,0, 0,2*lim,0, color='k', lw=1, arrow_length_ratio=0.05)   # y-axis
    ax.quiver(0,0,-lim, 0,0,2*lim, color='k', lw=1, arrow_length_ratio=0.05)   # z-axis
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)        # Box limits
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")                 # Axis Labels
ax1.set_title("Minimum configuration from initial sampling")
ax3.set_title("Minimum configuration from metropolis simulation")
fig.tight_layout(h_pad=3,w_pad=5)


tf = time()
print(f"\nProcess finished in {tf-to:.2f}s")
plt.show()

# Should return (kcal/mol): E=-6.765 Eelec=-8.57 VdW=1.805 ("true" minimum).
# Testing different values can bring better and more close results,
# but normally stays around E~-6.5 kcal/mol, which is a good approximation.
# The step size and angle, iterations, etc. may be ajusted.
#
# The acceptance depends quite strictly on the angle and step size, if in each
# metropolis step the angle is allowed to change much or a big step size is given, 
# the energy difference will be very high and few times will be accepted.
