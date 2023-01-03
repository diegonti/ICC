"""
Montecarlo Script to compute the most stable configuration of two water molecules.
Implementando Metropolis (dE)
(ADD: implement more molecules?)

Diego Ontiveros Cruz -- 10/11/2022
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

to = time()

def coulomb(qi,qj,r):
    """Returns coulomb energy between i-j pair, in kcal/mol."""
    kcal = 332.0 # To return the results in kcal/mol
    return kcal*qi*qj/r

def vdW(A,B,r):
    """Returns Van der Waals energy between i-j pair, , in kcal/mol."""

    return A/r**12 - B/r**6

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


# Energies parameters (VdW A and B, and coulomb charges)
# In matrix form since it will be easier and faster to computa later
# Each A,B and Q contain all the combinations ij of Aij,Bij and qij.
A = np.array([[581935.563838,328.317371,328.317371],
            [328.317371,9.715859e-6,9.715859e-6],
            [328.317371,9.715859e-6,9.715859e-6]])
B = np.array([[594.825035,10.478040,10.478040],
            [10.478040,0.001337,0.001337],
            [10.478040,0.001337,0.001337]])
q = np.array([-0.834,+0.417,+0.417])
Q = np.dot(q[:,None],q[:,None].T)

# Simulation parameters
T=300.              # Temperature
kb = 0.00119872041  # Boltzman constant (in kcal/(mol⋅K))
beta = 1./(kb*T)    # Beta factor


class Water():
    """
    A tip3p Water molecule class.
    """
    def __init__(self):
        self.rO = np.array([0.,0.,0.])
        self.rH1 = np.array([0.75669,0.58589,0.])
        self.rH2 = np.array([-0.75669,0.58589,0.])
        # self.coordOrigin = np.array([self.rO,self.rH1,self.rH2]).T
        self.coord = np.array([self.rO,self.rH1,self.rH2])
        
        

    def translate(self,Tx,Ty,Tz):
        """Translates the molecule to a specified point for each main coordinate."""
        self.coord = self.coord.T   # Transpose returns arrays by coordinate (X,Y,X) instead of (O,H1,H2)
        for i,T in enumerate([Tx,Ty,Tz]):
            self.coord[i] += T
        self.coord = self.coord.T
        ##Probar de hacerlo con arrays/broadcasting ??


    def rotate(self,Ax,Ay,Az, atype="deg"):
        """Rotates the molecule a specified angle (in degrees) for each main axis."""
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
                #R = np.dot(Rz,Ry,Rx) does not work 100%

        # Returns the molecule to the same place before (but rotated)
        self.translate(*temp) 

        ##Probar de hacerlo con arrays/broadcasting ?? o con R ??

    def toOrigin(self):
        """Moves the molecule to the origin of coorinated (centered in the Oxygen)"""
        c = self.coord[0]
        # print(c, "Test")
        self.translate(-c[0],-c[1],-c[2]) # test con temp\arriba

       
    def getEnergies(self,other):
        """Gets the energy of a pair of molecules."""
        coord1,coord2 = self.coord,other.coord

        diff = coord1-coord2[:,None]
        distances = np.sqrt((diff**2).sum(axis=-1)).T
        
        Evdw = (A/distances**12 - B/distances**6).sum()
        Eelec = (332.0*Q/distances).sum()
        E = Evdw + Eelec
        self.E = E
        return E,Eelec,Evdw


    def draw(self,ax):
        ax.scatter(self.coord.T[0],self.coord.T[1],self.coord.T[2], sizes=(100,75,75), c=("red","grey","grey"), alpha=True)


    def fixed(self): pass

    def reset(self): self.E = 0
    
    def _distances(self):
        """Test function to compute distances and see if is mantained rigid"""
        self.OH1 = np.linalg.norm(self.coord[1]-self.coord[0])
        self.OH2 = np.linalg.norm(self.coord[2]-self.coord[0])
        print(f"Computed: {self.OH1:.3f},{self.OH2:.3f}")
        print("Sould be: ", 0.957)

    def save(self):
        ## Ir guardando coordenadas y energias para poder comparar i, i-1
        pass

## Main Program      

# Opening visualization
fig = plt.figure(figsize=(10,7))
# spec = gridspec.GridSpec(ncols=2, nrows=2,
#                          width_ratios=[2, 1], wspace=0.5,
#                          hspace=0.5, height_ratios=[1, 2])

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, (3,4))
ax3 = fig.add_subplot(2, 2, 2, projection='3d')



# Main Montecarlo loop
n = 2           # Number of waters
N = 100000       # Number of iterations

# Initialization of fixed water molecule at center
wFixed = Water()
wFixed.draw(ax1)
wFixed.draw(ax3)


# List with the moving water molecules
waters = [Water() for _ in range(n-1)]

# Lists wher energies and coordinates will be saved
energies = [[] for _ in range(n-1)]
coords = [[] for _ in range(n-1)]

# Initial sampling of the box to get startig point
for j in range(N):
    
    # waters = [Water() for _ in range(n)]
    for i,w in enumerate(waters):

        # if j == 0:  randT = np.random.uniform(-5,5, size=3)
        # else:  randT = np.random.uniform(-5,5, size=3)
        
        # Random translation step and rotation
        randT = np.random.uniform(-5,5, size=3)     # To random position in a (-lim,lim) box
        randA = np.random.uniform(-180,180,size=3)  # With angles between (-180,180) degrees
        
        # Translation of waters
        w.translate(*randT)

        # Rotation of waters
        w.rotate(*randA,"d")


        # Energy calculations 
        E,Eelec,Evdw = w.getEnergies(wFixed)
        # print(f"E={E},Eelec={Eelec},Evwd={Evdw}")

        # Saving energy and coordinates to array
        ## Maybe save water object insetad?
        energies[i].append(E)
        coords[i].append(w.coord.copy())

        # Testing
        # w._distances()

        # Visualization of each step (NO!)
        # w.draw()

        # Resets energy and positions
        w.reset()
        w.toOrigin()

# Get minimum Energy and coordinates from the initial sampling 
energies = np.array(energies)
minEs = [np.min(e) for e in energies]
print("Minimum Energies from sampling: ", minEs)
minE = np.min(minEs)
minEi = np.argmin(minEs)
# print(coords[np.argmin(energies)])

##Esto podria mejorarse si en vez de guardar E y coords se guardase w.copy??
## Ojito con el paripe este de los indices --> Fuente error
coords = np.array(coords)
minCoords = []
for i in range(n-1):
    wMin = Water()
    minc = coords[i][np.argmin(energies[i])]
    wMin.coord = minc
    minCoords.append(minc)
    wMin.draw(ax1)

minCoords = np.array(minCoords)
wMin = Water()
wMin.coord = minCoords[minEi]

## Guardad energies del sampleado para hacer histograma?

# Lists wher energies and coordinates will be saved
energies2 = [minE]
coords2 = [minCoords[minEi]]

# Metroplolis from an energy-low initial point
# Takes the previously sampled minimum-energy water and uses that to start metropolis
N = 100
step = 0.001
acceptance = [0,0]
nPoints = [0]
w = wMin
for i in range(N):
    randT = np.random.uniform(-step,step, size=3)       # Random steps of (-step,step) lenghts
    randA = np.random.uniform(-90,90,size=3)              # With rotations between (-10,10) degrees
    
    # Translation of waters
    w.translate(*randT)

    # Rotation of waters
    w.rotate(*randA,"d")

    # Energy calculations 
    E,Eelec,Evdw = w.getEnergies(wFixed)
    # print(f"E={E},Eelec={Eelec},Evwd={Evdw}")

    dE = E - energies2[-1]

    # Acceptance of the step
    if not acceptE(dE):
        acceptance[1] += 1
        w.coord = coords2[-1]
        continue
    else: 
        acceptance[0] += 1

        # Saving energy and coordinates to array
        ## Maybe save water object insetad?
        energies2.append(E)
        coords2.append(w.coord.copy())
        nPoints.append(i+1)

    # Testing
    # w._distances()

    # Visualization of each step (NO!)
    # w.draw()

    # Resets energy and positions
    w.reset()
    # w.toOrigin()

print()
Emin = np.min(energies2)
coordMin = coords2[np.argmin(energies2)]
print(f"Accepted: {acceptance[0]}, Not Accepted: {acceptance[1]}. N_steps: {sum(acceptance)}")
print(f"Acceptance: {100*acceptance[0]/N:.2f}%")
print(f"Energy at final iteration: {energies2[-1]:.3f}")
print(f"Minimum energy found: {np.min(energies2):.3f}")

#Energy Plot settings
ax2.plot(nPoints,energies2, lw=1)
ax2.set_xlabel("N");ax2.set_ylabel("Energy (kcal/mol)")
ax2.set_title("Energy variation in MonteCarlo Metropolis")

# Min Configuration Plot
wMin = Water()
wMin.coord = coordMin
wMin.draw(ax3)


#Plot Settings
lim = 5 #box limit
for ax in [ax1,ax3]:
    ax.quiver(-lim,0,0, 2*lim,0,0, color='k', lw=1, arrow_length_ratio=0.05)   # x-axis
    ax.quiver(0,-lim,0, 0,2*lim,0, color='k', lw=1, arrow_length_ratio=0.05)   # y-axis
    ax.quiver(0,0,-lim, 0,0,2*lim, color='k', lw=1, arrow_length_ratio=0.05)   # z-axis
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)        # Box limits
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")                 # Axis Labels
ax1.set_title("Minimum configuration from initial sampling")
ax3.set_title("Minimum configuration from metropolis simulation")





#Deberia dar (kcal/mol): E=-6.765 Eelec=-8.57 VdW=1.805


tf = time()
print(f"Process finished in {tf-to:.2f}s")
fig.tight_layout(h_pad=3,w_pad=5)
plt.show()