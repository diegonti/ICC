### Jacobi diagonalization - Diego Ontiveros

import numpy as np

def maxElem(a):
    """Find larget off-diagonal elemen a[k,l]"""
    n = len(a)
    aMax = 0.0
    for i in range(n-1):
        for j in range(i+1,n):
            if abs(a[i,j]) >= aMax:
                aMax = abs(a[i,j])
                k = i; l = j
    return aMax,k,l

def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
    """Rotates matrix a to make a[k,l]=0"""
    n = len(a)
    aDiff = a[l,l] - a[k,k]
    if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
    else:
        phi = aDiff/(2.0*a[k,l])
        t = 1.0/(abs(phi) + np.sqrt(phi**2 + 1.0))
        if phi < 0.0: t = -t
    c = 1.0/np.sqrt(t**2 + 1.0); s = t*c
    tau = s/(1.0 + c)
    temp = a[k,l]
    a[k,l] = 0.0
    a[k,k] = a[k,k] - t*temp
    a[l,l] = a[l,l] + t*temp
    for i in range(k):      # Case of i < k
        temp = a[i,k]
        a[i,k] = temp - s*(a[i,l] + tau*temp)
        a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
    for i in range(k+1,l):  # Case of k < i < l
        temp = a[k,i]
        a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
        a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
    for i in range(l+1,n):  # Case of i > l
        temp = a[k,i]
        a[k,i] = temp - s*(a[l,i] + tau*temp)
        a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
    for i in range(n):      # Update transformation matrix
        temp = p[i,k]
        p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
        p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

def jacobi(a,tol = 1.0e-9):
    """ 
    Solution of eigenvalue problem [A]{p} = eval{p}
    by Jacobi's method. Returns eigenvalues in vector eval, 
    and the unitary transformation matrix p.

    p contains the eigenvectors as columns of the matrix.

    eval,p = jacobi(A,tol = 1.0e-9).
    """
    n = len(a)
    maxRot = 10*(n**2)      # Set limit on number of rotations
    p = np.identity(n)*1.0     # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop 
        aMax,k,l = maxElem(a)
        if aMax < tol: 
            return np.diagonal(a),p
        rotate(a,p,k,l)
    print ('Jacobi method did not converge')

# Input matrix
A = np.array([[1.,5.,0.,5],
           [5.,1.+25.,5.,0.],
           [0.,5.,3.-5.,1.],
           [5.,0.,1.,9.-5.]])

# PAP and QAQ quadrants
N = len(A)
PAP = A[:int(N/2),:int(N/2)]
QAQ = A[int(N/2):,int(N/2):]

print("Input matrix:")
print(A)

# Obtainint eigenvalues (eval) and transformation matrix (p)
# from the jacobi method 
eval,p = jacobi(A.copy()) 
print("\nEigenvalues:")
print(eval)
print("\nTransformation matrix:")
print(p)
print("\nDoes A@p = eval*p ? (eigenvalue problem): ",np.all(np.isclose((A@p),(eval*p))))

# Effective Hamiltonian (p.T@A@p)
print("\nDiagonal matrix (p.T@A@p):")
H_eff = p.T@A@p
print(H_eff)

tol = 1e-12
print(f"\nAfter rounding values < {tol} to zero:")
H_eff_rounded = H_eff.copy()
H_eff_rounded[np.abs(H_eff_rounded) < tol] = 0
print(H_eff_rounded)

