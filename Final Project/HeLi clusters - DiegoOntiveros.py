"""
Montecarlo Script to compute the most stable configuration of He-Li clusters.
The program takes N He atoms and a single centered Li atom and uses the metropolis
MC algorithm to find the optimized structure.

It uses a first sampling of the systems box to find stable points, and from
there, starts the metropolis algorithm to find the energy minimum.

Results 

For more info, see also the testingMC.ipynb notebook, were different tests were performed,
(scaling and velocity tests, functions tests, visualization, etc).

Diego Ontiveros Cruz -- 10/1/2023
"""

import numpy as np
import matplotlib.pyplot as plt


from itertools import combinations
from copy import deepcopy
from time import time
to = time()

#################### General external functions ##################

def acceptE(dE,beta):
    """
    Accepts or regects a Monte Carlo step.

    Parameters
    ----------
    `delta_e` : Difference in energy from one step to another.
    `beta` : Temperature factor to consider.

    Returns
    ----------
    `True` : if step is accepted. 
    `False` : if step is not accepted.      
    """
    if dE < 0: 
        return True
    else:
        rand = np.random.random()
        if np.exp(-beta*dE) > rand:
            return True
        else:
            return False
            
def print_title(text,before=15,after=15,separator="-",head=2,tail=1, file_name:str=None):
    """Prints text in a title-style."""
    separator = str(separator)
    print("\n"*head,separator*before,text,separator*after,"\n"*tail)

    if file_name is not None:
        with open(file_name,"a") as outFile:
            print("\n"*head,separator*before,text,separator*after,"\n"*(tail-1),file=outFile)


###################### He-He and He-Li Potentials ############### 

def fnHeLi(R,n,b):
    suma = np.sum([(b*R)**k /np.math.factorial(k) for k in range(n+1)],axis=0)
    # k = np.arange(n+1)
    # suma = np.sum( (b*R)**k/sp.special.factorial(k) ) # broadcastin!
    return 1 - np.exp(-b*R) * suma

def fnHeHe(x,D):
    return np.exp(-(D/x-1)**2)


def HeLiPotential(R):
    """
    Computes the He-Li potential asi in Chem. Phys. Lett. (2001), 343, 429-436.

    Parameters
    ----------
    `R` : Distance between He-Li particles.

    Returns
    -------
    `V` He-Li potential in cm-1.
    """

    # Faster power calculations
    R2 = R*R
    R4 = R2*R2
    R6 = R4*R2
    R7 = R6*R
    R8 = R4*R4

    # Parametrized coefitients
    A,b = 20.8682, 2.554
    D4 = 1.383192/2
    D6 = 2.4451/2 + 0.298
    D7 = 7.3267/2
    D8 = 10.6204/2 + 43.104/24 + 1.98
    
    V4 = fnHeLi(R,4,b)*D4/R4
    V6 = fnHeLi(R,6,b)*D6/R6
    V7 = fnHeLi(R,7,b)*D7/R7
    V8 = fnHeLi(R,8,b)*D8/R8

    V = A*np.exp(-b*R) - V4 - V6 - V7 - V8
    cm = 219474.63
    return V*cm

def HeHePotential(R):
    """
    Computes the LM2M2 He-He potential.

    Parameters
    ----------
    `R` : Distance between He particles.

    Returns
    -------
    `V` : He-He potential in cm-1.
    """
    eps = 10.97
    rm = 2.9695/0.529177
    x = R/rm

    # Faster power calculations
    x2 = x*x
    x6 = x2*x2*x2
    x8 = x6*x2
    x10 = x8*x2

    # Parametrized coefitients
    A = 1.89635353e5
    a,b = 10.70203539, -1.90740649
    D6,D8,D10 = 1.4651625, 1.2, 1.1
    c6, c8, c10 = 1.34687065, 0.41308398, 0.17060159

    f6 = fnHeHe(x,D6)
    f8 = fnHeHe(x,D8)
    f10 = fnHeHe(x,D10)

    Vx = A*np.exp(-a*x+b*x**2) - f6*c6/x6 -f8*c8/x8 - f10*c10/x10 
    V = eps*Vx
    cm = 219474.63
    K = 315777
    return V*cm/K

def potential(atom1,atom2,R):
    HeLi = isinstance(atom1,He) and isinstance(atom2,Li)
    LiHe = isinstance(atom1,Li) and isinstance(atom2,He)
    HeHe = isinstance(atom1,He) and isinstance(atom2,He)

    if HeLi or LiHe: V = HeLiPotential(R)
    elif HeHe: V = HeHePotential(R)

    return V



#################### System and Atoms classes ###############

class Li():
    """ Li particle class. """
    def __init__(self) -> None:
        self.coord = np.zeros(3)
        self.index = 1
        self.label = "Li"

    def draw(self,ax:plt.Axes):
        """Draws Li atom object to a specified axis."""
        x,y,z = self.coord
        ax.scatter(x,y,z, s=100, c="forestgreen", alpha=0.75, edgecolors="k")

class He():
    """ He particle class. """
    def __init__(self,index:int) -> None:
        self.coord = np.random.uniform(-1,1,size=3)
        self.index = index
        self.label = "He"

    def translate(self,Tx,Ty,Tz):
        """Translates the atom object a specified amount for each main coordinate.
        
        Parameters
        ----------
        `Tx,Ty,Tz` : Translation step on the x,y,z coordinates, respectively.
        
        """
        self.coord += np.array([Tx,Ty,Tz])

    def toOrigin(self):
        """Moves the atom to the origin of coorinated."""
        self.coord = np.zeros(len(self.coord))

    def draw(self,ax:plt.Axes):
        """Draws He atom object to a specified axis."""
        x,y,z = self.coord
        ax.scatter(x,y,z, s=100, c="darkorchid", alpha=0.75, edgecolors="k")

class System():
    def __init__(self, atoms:list) -> None:
        """
        Contains information of the systsem in pair-whise manner.

        Parameters
        ----------
        `atoms` : List of atoms of particle objects.
        """
        self.atoms = atoms
        self.pairs = list(combinations(atoms,2))    # List of pairs of particles objects
        self.N = len(self.atoms)                    # Number of particles

        # Pairs information
        self.labels = self.get_labels()
        self.distances = self.get_distances()
        self.energies = self.get_energies()
        self.total_energy = self.get_total_energy()


    def get_labels(self):
        """Gets the atom labels for each pair. Returns list."""

        self.labels = []
        for pair in self.pairs:
            self.labels.append((pair[0].label+str(pair[0].index),pair[1].label+str(pair[1].index)))
        return self.labels

    def get_distances(self):                                                          
        """Gets the distances between atoms of each pair. Returns array."""

        self.distances = []
        for pair in self.pairs:
            distance = np.linalg.norm(pair[0].coord - pair[1].coord)
            self.distances.append(distance)

        return np.array(self.distances)

    def get_energies(self):
        """Gets the energies between atoms of each pair. Returns array."""

        self.energies = []
        for i,pair in enumerate(self.pairs):
            atom1,atom2 = pair[0],pair[1]
            R = self.distances[i]
            V = potential(atom1,atom2,R)
            self.energies.append(V)
        return np.array(self.energies)

    def get_total_energy(self):
        """Computes total energy of the system by the sum of energies of each pair."""

        self.total_energy = np.sum(self.energies)
        return self.total_energy

    def update(self):
        """Updates all attributes of the System."""
        self.labels = self.get_labels()
        self.distances = self.get_distances()
        self.energies = self.get_energies()
        self.total_energy = self.get_total_energy()


    def draw(self,ax:plt.Axes):
        """Draws the atoms of the system to the specified axis"""
        for atom in self.atoms:
            atom.draw(ax)

    def writeXYZ(self,file_name:str):
        """
        Writes the coords array into a .xyz-style file.

        Parameters
        ----------
        `file_name` : Output file name (.xyz). Defaults to "MCout.xyz"
        """
        N = self.N
        with open(file_name,"w") as outFile: 
            outFile.write(str(N) + "\n\n")
            for atom in self.atoms:
                label = atom.label
                x,y,z = atom.coord
                outFile.write(f"{label:5} {x:>20.15f} {y:>20.15f} {z:>20.15f} \n")


########################### ----- Main Program ----- #########################      
##############################################################################

# Opening visualization and creating figure axes
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, (3,4))
ax3 = fig.add_subplot(2, 2, 2, projection='3d')


# Input parameters (Initial Sampling and Metropolis)
N_He = 6                # Number of He atoms 
lim = 8                 # Box limit
N_sampling = 50000      # Number of sampling iterations
N_metropolis = 200000   # Number of metropolis iterations 
step = 0.1              # Size of the translatio step in metropolis
T = 10.                 # Temperature
file_name = "out.log"   # Output file name

kb = 0.695034800        # Boltzman constant (in cm-1/K)
beta = 1./(kb*T)        # Beta factor

print(f"\nSystem: He{N_He}Li+")
print("Temperature (K): ", T)
print("Number of initial sampling steps: ", N_sampling)
print("Number of Metropolis MC steps:    ", N_metropolis)

with open(file_name,"w") as outFile:
    outFile.write(f"\nSystem: He{N_He}Li+\n")
    outFile.write(f"Temperature (K): {T}\n")
    outFile.write(f"Number of initial sampling steps: {N_sampling}\n")
    outFile.write(f"Number of Metropolis MC steps:    {N_metropolis}\n")


# Creating list with the atom objects of the system
atoms = [Li()]
for i in range(N_He): atoms.append(He(i+1))

# Creating System with the atoms
system = System(atoms)


######################## ----- Initial MC Sampling ----- #########################  

"""
Initial sampling of the box to get a startig point for metropolis.
While the Li+ is fixed in the center, snapshots of the He atoms at random places 
are generated and the energy is saved.
"""

print_title("Starting initial sampling",file_name=file_name)
print("Completed:", end=" ")
frames:list[System] = [0 for _ in range(N_sampling)]
energies = np.zeros(N_sampling) 
for i in range(N_sampling):

    # Generating random position for the He atoms in the box
    for atom in system.atoms[1:]:
        randT = np.random.uniform(-lim,lim,size=3)
        atom.translate(*randT)
    
    system.update()                                 # Updating system parameters
    frames[i] = deepcopy(system)                    # Saving system snapshot
    energies[i] = system.total_energy               # Saving total energy
    for atom in system.atoms[1:]: atom.toOrigin()   # Moving He to origin

    # % Completed
    if i%(N_sampling/10) == 0:
        print(f"{int(100*i/N_sampling)}% ",sep=" ",end="",flush=True)


# Get minimum Energy and Coordinates from the initial sampling 
minE = np.min(energies)                 # Mininum energy from sampling pairs
minEi = np.argmin(energies)             # Minimum energy index
minFrame = frames[minEi]                # Frame of the minimum energy 
minFrame.draw(ax1)                      # Drawing initial sample minimum    
minFrame.writeXYZ(file_name="sampling.xyz") 
print(f"\nMinimum Energies from sampling (cm-1): {minE:.6f}")
with open(file_name,"a") as outFile: outFile.write(f"\nMinimum Energies from sampling (cm-1): {minE:.6f}\n")



######################## ----- Metropolis MC ----- #########################   

"""
Metropolis Monte Carlo to find the minimum configuration from the initial starting point
found in the sampling. Now the He atoms move a smaller given random step,
and new configurations are gathered through the Metropolis algorithm.
"""

nPoints = [0]                   # List of accepted steps
acceptance = [0,0]              # MC Metropolis acceptance 
energies = [minE]               # List of energies
system = deepcopy(minFrame)
frames = [deepcopy(minFrame)]

print_title("Starting Metropolis MC",file_name=file_name)
print("Completed:", end=" ")
for i in range(N_metropolis):

    # Random steps for the He atoms
    for atom in system.atoms[1:]: 
        randT = np.random.uniform(-step,step,size=3)
        atom.translate(*randT)    
    
    system.update()
    E = system.total_energy
    dE = E - energies[-1]

    # Acceptance of the MC step
    if acceptE(dE,beta):
        # Save energies and snapshot
        acceptance[0] += 1
        energies.append(E)
        nPoints.append(i+1)
        frames.append(deepcopy(system))
    else:
        # Turn back to last frame
        acceptance[1] += 1
        system = deepcopy(frames[-1])

    # % Completed
    if i%(N_metropolis/10) == 0:
        print(f"{int(100*i/N_metropolis)}% ",sep=" ",end="",flush=True)


minE = np.min(energies)             # Minimum energy from metropolis MC
minEi = np.argmin(energies)         # Minimum energy index
minFrame = frames[minEi]            # Minimum frame for the system
minFrame.draw(ax3)                  # Drawing minimum configuration
ax2.plot(nPoints,energies, lw=1)    # Plotting energy evolution


minFrame.writeXYZ(file_name="metropolis.xyz")

print(f"\nAccepted: {acceptance[0]}, Not Accepted: {acceptance[1]}. N steps: {sum(acceptance)}")
print(f"Acceptance:                              {100*acceptance[0]/N_metropolis:.2f}%")
print(f"Energy at final iteration (cm-1):    {energies[-1]:.6f}")
print(f"Minimum energy found (cm-1):         {minE:.6f}")

with open(file_name,"a") as outFile:
    outFile.write(f"\nAccepted: {acceptance[0]}, Not Accepted: {acceptance[1]}. N steps: {sum(acceptance)}\n")
    outFile.write(f"Acceptance:                              {100*acceptance[0]/N_metropolis:.2f}%\n")
    outFile.write(f"Energy at final iteration (cm-1):    {energies[-1]:.6f}\n")
    outFile.write(f"Minimum energy found (cm-1):         {minE:.6f}\n")



# Plot titles and axis labels
ax1.set_title("Minimum configuration from initial sampling")
ax3.set_title("Minimum configuration from metropolis simulation")
ax2.set_title("Energy variation in MonteCarlo Metropolis")
ax2.set_xlabel("N");ax2.set_ylabel("Energy (cm-1)")

# 3D Plots axis arrows
tick_step = 4
for ax in [ax1,ax3]:
    ax.quiver(-lim,0,0, 2*lim,0,0, color='k', lw=1, arrow_length_ratio=0.05)   # x-axis
    ax.quiver(0,-lim,0, 0,2*lim,0, color='k', lw=1, arrow_length_ratio=0.05)   # y-axis
    ax.quiver(0,0,-lim, 0,0,2*lim, color='k', lw=1, arrow_length_ratio=0.05)   # z-axis
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)        # Box limits
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")                 # Axis Labels
    ax.set_xticks(np.arange(-lim, lim+tick_step, tick_step))
    ax.set_yticks(np.arange(-lim, lim+tick_step, tick_step))
    ax.set_zticks(np.arange(-lim, lim+tick_step, tick_step))


fig.tight_layout(h_pad=3,w_pad=5)
fig.savefig(f"He{N_He}Li.jpg",dpi=600)


tf = time()
print(f"\nProcess finished in {tf-to:.2f}s\n")
with open(file_name,"a") as outFile: outFile.write(f"\nProcess finished in {tf-to:.2f}s\n")
plt.show()


