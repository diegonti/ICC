### Lanczos diagonalization - Diego Ontiveros

import numpy as np
import numpy.linalg as la

#Generates the representation in a subspace of a Hermitian NxN matrix using the Lanczos algorithm and an initial vector guess vg.
def lanczos(H,vg):
    Lv=np.zeros((len(vg),len(vg)), dtype=complex) #Creates matrix for Lanczos vectors
    Hk=np.zeros((len(vg),len(vg)), dtype=complex) #Creates matrix for the Hamiltonian in the subspace
    Lv[0]=vg/la.norm(vg) #Creates the first Lanczos vector as the normalized guess vector vg
     
    #Performs the first iteration step of the Lanczos algorithm
    w=np.dot(H,Lv[0]) 
    a=np.dot(np.conj(w),Lv[0])
    w=w-a*Lv[0]
    Hk[0,0]=a
     
    #Performs the iterative steps of the Lanczos algorithm
    for j in range(1,len(vg)):
        b=(np.dot(np.conj(w),np.transpose(w)))**0.5
        Lv[j]=w/b
         
        w=np.dot(H,Lv[j])
        a=np.dot(np.conj(w),Lv[j])
        w=w-a*Lv[j]-b*Lv[j-1]
        
        #Creates tridiagonal matrix Hk using a and b values
        Hk[j,j]=a
        Hk[j-1,j]=b
        Hk[j,j-1]=np.conj(b)
        
    return (Hk,Lv)

def QR_decomposition(A):
    """Performs a QR decomposition step"""
    n, m = A.shape # get the shape of A

    Q = np.empty((n, n)) # initialize matrix Q
    u = np.empty((n, n)) # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R

def QR_eigvals(A, tol=1e-12, maxiter=1000):
    "Find the eigenvalues of A using QR decomposition."

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = QR_decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals

# Input matrix
A = np.array([[1.,5.,0.,5],
           [5.,1.+25.,5.,0.],
           [0.,5.,3.-5.,1.],
           [5.,0.,1.,9.-5.]])

print("\nInput matrix:")
print(A)

v = np.random.random(4)
T,V = lanczos(A.copy(),v)


print("\nTridiagonal matrix:")
print(T.real)

eig = QR_eigvals(T.real)
print("\nEigenvalues:")
print(eig)

