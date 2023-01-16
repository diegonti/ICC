import os

for N_He in range(11,20+1):
    path = f"He{N_He}Li/"
    os.rename(path+f"metropolis{N_He}.xyz",path+f"metropolis.xyz")
