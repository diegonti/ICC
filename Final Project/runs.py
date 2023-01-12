from HeLi_clusters import He,Li,System,MonteCarlo
from HeLi_clusters import print_title

from time import time
import os
to = time()

# Input parameters (Initial Sampling and Metropolis)
lim = 8                 # Box limit
N_sampling = 5000      # Number of sampling iterations
N_metropolis = 2000   # Number of metropolis iterations 
step = 0.1              # Size of the translatio step in metropolis
T = 10.                 # Temperature

dir_runs = "runs2"
try: os.mkdir(dir_runs)
except FileExistsError: pass

for N_He in range(11,13):
    
    # file_name = f"out{N_He}.log"  # Output file name
    dir_name = f"./{dir_runs}/He{N_He}Li/"

    try: os.mkdir(dir_name)
    except FileExistsError: pass


    # Creating list with the atom objects of the system
    atoms = [Li()]
    for i in range(N_He): atoms.append(He(i+1))

    # Creating System with the atoms
    system = System(atoms)

    # Setting and running the MC simulation
    mc = MonteCarlo(
        system,
        N_sampling,N_metropolis,
        T,step,lim,
        path=dir_name

    )
    mc.run()

tf = time()
print(f"\n\nTotal process finished in {tf-to:.2f}s\n")