"""
Montecarlo Script to compute the most stable configuration of two water molecules.
Solo con random sampling, no metroplolis (comparando dE)

Diego Ontiveros Cruz -- 6/11/2022
"""
import numpy as np
import matplotlib.pyplot as plt

def coulomb(qi,qj,r):
    """Returns coulomb energy between i-j pair."""
    kcal = 332.0 #To return the results in kcal/mol
    return kcal*qi*qj/r

def vdW(A,B,r):
    """Returns Van der Waals energy between i-j pair."""
    # eps,ro = 1,3
    #epsij = sqrt(epsi*epsj)
    #roij = roi+roj
    # A,B = eps*ro**12, 2*eps*ro**6 
    return A/r**12 - B/r**6

##Opt: diccionario con eij rij generales y cojer segun indice/label
#Energies parameters (VdW A and B, and coulomb charges)
A = np.array([[581935.563838,328.317371,328.317371],
            [328.317371,9.715859e-6,9.715859e-6],
            [328.317371,9.715859e-6,9.715859e-6]])
B = np.array([[594.825035,10.478040,10.478040],
            [10.478040,0.001337,0.001337],
            [10.478040,0.001337,0.001337]])
q = np.array([-0.834,+0.417,+0.417])

#Simulation parameters
T=300.              #Temperature
kb = 0.00119872041  #Boltzman constant (in kcal/(mol⋅K))
beta = 1./(kb*T)    #Beta factor


class Water():
    def __init__(self):
        qO = -0.834
        qH = +0.417
        self.rO = np.array([0.,0.,0.])
        self.rH1 = np.array([0.75669,0.58589,0.])
        self.rH2 = np.array([-0.75669,0.58589,0.])
        # self.coordOrigin = np.array([self.rO,self.rH1,self.rH2]).T
        self.coord = np.array([self.rO,self.rH1,self.rH2])
        
        

    def translate(self,Tx,Ty,Tz):
        """Translates the molecule to a specified point for each main coordinate."""
        self.coord = self.coord.T   #Transpose returns arrays by coordinate (X,Y,X) instead of (O,H1,H2)
        for i,T in enumerate([Tx,Ty,Tz]):
            self.coord[i] += T
        self.coord = self.coord.T


    def rotate(self,Ax,Ay,Az, atype="deg"):
        """Rotates the molecule a specified angle (in degrees) for each main axis."""
        atype = atype.lower()
        if atype == "deg" or atype == "d": Ax,Ay,Az = np.radians(Ax),np.radians(Ay),np.radians(Az)
        elif atype == "rad" or atype == "r": pass
        else: raise TypeError("Angle type not detected. Choose deg for degrees or rad for radians.")

        #Rotation matrices
        Rx = np.array([[1,0,0],[0,np.cos(Ax),-np.sin(Ax)],[0,np.sin(Ax),np.cos(Ax)]])
        Ry = np.array([[np.cos(Ay),0,np.sin(Ay)],[0,1,0],[-np.sin(Ay),0,np.cos(Ay)]])
        Rz = np.array([[np.cos(Az),-np.sin(Az),0],[np.sin(Az),np.cos(Az),0],[0,0,1]])

        #Moves to origin #OHH
        temp = [self.coord[0][i] for i in range(3)] #Gets temporal coordinates of Oxygen, to later replace it on the same spot it was
              
        self.toOrigin()

        #Rotates the molecule the specified angles
        for i in range(3):
            for _,R in enumerate([Rx,Ry,Rz]):
                ctemp = self.coord[i].copy()
                self.coord[i] = np.dot(R,ctemp) 
                #R = np.dot(Rz,Ry,Rx)

        #Returns the molecule to the same place before (but rotated)
        self.translate(*temp) 

    def toOrigin(self):
        """Moves the molecule to the origin of coorinated (centered in the Oxygen)"""
        c = self.coord[0]
        # print(c, "Test")
        self.translate(-c[0],-c[1],-c[2]) #test con temp\arriba

       
    def getEnergies(self,other):
        """Gets the energy of a pair of molecules."""
        coord1,coord2 = self.coord,other.coord

        E,Eelec,Evdw = 0,0,0
        for i in range(3):      #For each atom in molecule 1
            for j in range(3):  #For each atom in molecule 2
                r = np.linalg.norm(coord2[j]-coord1[i])   
                #Calculadas a pares de c[1]-c[2] (no energias interas) 
                Evdw += vdW(A[i][j],B[i][j],r)
                Eelec += coulomb(q[i],q[j],r)
        E = Eelec+Evdw  
        self.E = E
        return E,Eelec,Evdw


    def draw(self):
        ax.scatter(self.coord.T[0],self.coord.T[1],self.coord.T[2], sizes=(100,75,75), c=("red","grey","grey"), alpha=True) #Fixed


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
##Main Program      

#Opening visualization
fig = plt.figure()
ax = plt.axes(projection="3d")

#Main Montecarlo loop
n = 2       #Number of waters
N = 10000       #Number of iterations

#inicializar un agua fuera fija, y el resto iterar
wFixed = Water()
wFixed.draw()
waters = [Water() for _ in range(n-1)]

energies = [[] for _ in range(n-1)]
coords = [[] for _ in range(n-1)]
for j in range(N):
    
    # waters = [Water() for _ in range(n)]
    for i,w in enumerate(waters):
        if j == 0:  randT = np.random.uniform(-5,5, size=3)
        else:  randT = np.random.uniform(-5,5, size=3)

        randA = np.random.uniform(-180,180,size=3)
        

        # print("Initial:")
        # print("O:",w.coord[0])
        # print("Translation: ", randT)
        w.translate(*randT)
        # print("After T:")
        # print("O:",w.coord[0])
        # print()


        # print("Rotation: ", randA)
        # print("O:",w.coord[0])
        w.rotate(*randA,"d")
        # print("After R:")
        # print(w.coord)
        # print()

        E,Eelec,Evdw = w.getEnergies(wFixed)
        # print(f"E={E},Eelec={Eelec},Evwd={Evdw}")
        energies[i].append(E)
        # print(w.coord)
        coords[i].append(w.coord.copy())
        # print(coords)


        # w._distances()

        #Visualization
        # w.draw()
        w.reset()
        w.toOrigin()

energies = np.array(energies)
print([np.min(e) for e in energies])
coords = np.array(coords)
# print(coords[np.argmin(Energies)])
for i in range(n-1):
    wMin = Water()
    wMin.coord = coords[i][np.argmin(energies[i])]
    wMin.draw()



#Plot Settings
lim = 5 #box limit
ax.quiver(-lim, 0, 0, 2*lim, 0, 0, color='k',lw=1, arrow_length_ratio=0.05) # x-axis
ax.quiver(0, -lim, 0, 0, 2*lim, 0, color='k',lw=1, arrow_length_ratio=0.05) # y-axis
ax.quiver(0, 0, -lim, 0, 0, 2*lim, color='k',lw=1, arrow_length_ratio=0.05) # z-axis
ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)           # Box limits
ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")                    # Axis Labels
plt.show()


#Deberia dar (kcal/mol): E=-6.765 Eelec=-8.57 VdW=1.805

#montecarlo: 
#   if Enew<Eold:
#       new random coordinates
#   elif Enew>Eold: metropolis
#       if alpha(0,1) < exp(-beta*dE) 
#       (alpha = random.uniform(0,1), dE=Enew-Eold)