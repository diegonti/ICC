import numpy as np
import matplotlib.pyplot as plt

rO = np.array([0.,0.,0.])
rH1 = np.array([0.75669,0.58589,0.])
rH2 = np.array([-0.75669,0.58589,0.])
coord = np.array([rO,rH1,rH2]).T

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(coord[0],coord[1],coord[2], sizes=(100,100,100), c=("red","grey","grey"), alpha=True) #Fixed
ax.scatter(coord[0]+2,coord[1]+2,coord[2]+2, sizes=(100,100,100), c=("red","grey","grey"), alpha=True) #Ir generando nueva geom


lim = 5
ax.quiver(-lim, 0, 0, 2*lim, 0, 0, color='k',lw=1, arrow_length_ratio=0.05) # x-axis
ax.quiver(0, -lim, 0, 0, 2*lim, 0, color='k',lw=1, arrow_length_ratio=0.05) # y-axis
ax.quiver(0, 0, -lim, 0, 0, 2*lim, color='k',lw=1, arrow_length_ratio=0.05) # z-axis
ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_zlim(-lim,lim)
ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
plt.show()