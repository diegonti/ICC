"""
Montecarlo Script to compute the most stable configuration of two water molecules.

Diego Ontiveros Cruz -- 6/11/2022
"""
import numpy as np
import matplotlib.pyplot as plt

def coulomb(qi,qj,r):
    return qi*qj/r

def vdW(r):
    eps,ro = 1,3
    #epsij = sqrt(epsi*epsj)
    #roij = roi+roj
    A,B = eps*ro**12, 2*eps*ro**6 
    return A/r**12 + B/r**6

VdW_paramA = np.array([[581935.563838,328.317371],[328.317371,9.715859e-6]])
VdW_paramB = np.array([[594.825035,10.478040],[10.478040,0.001337]])


class Water():
    def __init__(self):
        qO = -0.834
        qH = +0.417
        self.rO = np.array([0.,0.,0.])
        self.rH1 = np.array([0.75669,0.58589,0.])
        self.rH2 = np.array([-0.75669,0.58589,0.])
        # self.coordOrigin = np.array([self.rO,self.rH1,self.rH2]).T
        self.coord = np.array([self.rO,self.rH1,self.rH2]).T   #Transpose returns arrays by coordinate (X,Y,X) instead of (O,H1,H2)
        ##Quizas poner coord normal y cambiar .T en menos sitios (plot)
        
        

    def translate(self,Tx,Ty,Tz):
        """Translates the molecule to a specified point for each main coordinate."""
        for i,T in enumerate([Tx,Ty,Tz]):
            self.coord[i] += T


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

        #Moves to origin
        temp = [self.coord[i][0] for i in range(3)] #Gets temporal coordinates of Oxygen, to later replace it on the same spot it was
        self.toOrigin()

        #Rotates the molecule the specified angles
        self.coord = self.coord.T
        for i in range(3):
            for _,R in enumerate([Rx,Ry,Rz]):
                ctemp = self.coord[i].copy()
                self.coord[i] = np.dot(R,ctemp) 
                #R = np.dot(Rz,Ry,Rx)
        self.coord = self.coord.T

        #Returns the molecule to the same place before (but rotated)
        self.translate(*temp) 

    def toOrigin(self):
        """Moves the molecule to the origin of coorinated (centered in the Oxygen)"""
        self.coord = self.coord.T
        self.translate(-self.coord[0],-self.coord[0],-self.coord[0]) #test con temp\arriba
        self.coord = self.coord.T

       
    def getEnergies(self,other):
        """Gets the VanDerWaals energy of a molecule with another."""
        coord1,coord2 = self.coord.T,other.coord.T

        E = 0
        for i in range(3):
            for j in range(3):
                r = np.linalg.norm(coord1[j]-coord2[i])    
                E += vdW(r) #E += vdW(A[i][j], B[i][j] ,r)
                #E += coulomb(q[i],q[j],r)
        pass

    def draw(self):
        ax.scatter(self.coord[0],self.coord[1],self.coord[2], sizes=(100,100,100), c=("red","grey","grey"), alpha=True) #Fixed

    def fixed(self): pass
    
    def _distances(self):
        """Test function to compute distances and see if is mantained rigid"""
        self.coord = self.coord.T
        self.OH1 = np.linalg.norm(self.coord[1]-self.coord[0])
        self.OH2 = np.linalg.norm(self.coord[2]-self.coord[0])
        print(f"Computed: {self.OH1:.3f},{self.OH2:.3f}")
        print("Sould be: ", 0.957)
        self.coord = self.coord.T

##Main Program      

#Opening visualization
fig = plt.figure()
ax = plt.axes(projection="3d")

#Main Montecarlo loop
n = 2       #Number of waters
N = 3       #Number of iterations
waters = [Water() for _ in range(n)]
for j in range(N):
    for i,w in enumerate(waters):
        if i != 0:
            randT = np.random.uniform(-5,5, size=3)
            randA = np.random.uniform(-180,180,size=3)
            print(randT)
            print(randA)
            w.translate(*randT)
            w.rotate(*randA,"d")
            
        elif j == 0: w.fixed()      #El primer caso de la agua central
        else: continue              #Los casos repetidos 

        w._distances()

        #Visualization
        w.draw()
        # ax.scatter(w.coord[0],w.coord[1],w.coord[2], sizes=(100,100,100), c=("red","grey","grey"), alpha=True)
        w.toOrigin()

#Plot Settings
lim = 5 #box limit
ax.quiver(-lim, 0, 0, 2*lim, 0, 0, color='k',lw=1, arrow_length_ratio=0.05) # x-axis
ax.quiver(0, -lim, 0, 0, 2*lim, 0, color='k',lw=1, arrow_length_ratio=0.05) # y-axis
ax.quiver(0, 0, -lim, 0, 0, 2*lim, color='k',lw=1, arrow_length_ratio=0.05) # z-axis
ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)           # Box limits
ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")                    # Axis Labels
plt.show()

# w = Water()
# print(w.coord)
# w.translate(2,-1,2)
# w.rotate(-90,0,0,"d")
# print(w.coord)


#1 agua fijada en el centero (xy), la otra va trasladandose y rotando con MC, 
# generar R,A aleatorios (saltos) y cojer la que es menor

# Las moleculas seran rjidas (r,a internos fijos) --> hacer todo en funcion de CM y rotacion de ejes principales (inercia)
# funciones --> getCM, translacion (desplacamientos random del CM), rotcoordenadas
#random.uniform(-0.5,0.5)

#Rotacion local (en centro de coordenadas)