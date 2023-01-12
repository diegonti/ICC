"""
Problem Extra - Eigenstates of atoms in harmonic trap.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

rang = 10
E = np.linspace(-rang,rang, 1000)
a = (1/2**0.5)*sp.special.gamma(-E/2+1/4)/sp.special.gamma(-E/2+3/4)

plt.plot(a,E, color = "red")
plt.axvline(0, ls="--", color="black", lw = 0.5)
plt.xlim(-8,8); plt.ylim(-rang,rang)
plt.xlabel(r"$a_o$"); plt.ylabel("E")

plt.show()



