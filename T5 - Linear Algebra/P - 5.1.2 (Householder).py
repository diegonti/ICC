"""
Problem 5.1.2 - Diagonalization.
HouseHolder method to diagonalize matrix.
Diego Ontiveros
"""
import numpy as np

def householder(A):
    """Returns the tridiagonal matrix from a given symmetric matrix."""

    n = A.shape[0]
    v = np.zeros(n, dtype=np.double)
    u = np.zeros(n, dtype=np.double)
    z = np.zeros(n, dtype=np.double)

    for k in range(0,n-2):

        if np.isclose(A[k+1,k], 0.0):
            a = -np.sqrt(np.sum(A[(k+1):,k]**2))
        else:
            a = -np.sign(A[k+1,k]) * np.sqrt(np.sum(A[(k+1):,k]**2))

        two_r_squared = a**2 - a*A[k+1,k]
        v[k] = 0.0
        v[k+1] = A[k+1,k] - a
        v[(k+2):] = A[(k+2):,k]
        u[k:] = 1.0 / two_r_squared * np.dot(A[k:,(k+1):], v[(k+1):])
        z[k:] = u[k:] - np.dot(u, v) / (2.0 * two_r_squared) * v[k:]

        for l in range(k+1, n-1):

            A[(l+1):,l] = (A[(l+1):,l] - v[l] * z[(l+1) :] - v[(l+1) :] * z[l])
            A[l,(l+1):] = A[(l+1):,l]
            A[l,l] = A[l,l] - 2*v[l]*z[l]

        A[-1,-1] = A[-1,-1] - 2*v[-1]*z[-1]
        A[k,(k+2):] = 0.0
        A[(k+2):,k] = 0.0

        A[k+1,k] = A[k+1,k] - v[k+1]*z[k]
        A[k,k+1] = A[k+1,k]

    return A

def QR_Decomposition(A):
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
        Q, R = QR_Decomposition(A_old)

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

HH_TD = householder(A.copy())
print("\nHouse Holder Tridiagonal matrix:")
print(HH_TD)

eig = QR_eigvals(HH_TD.copy())
print("\nEigenvalues: ")
print(eig)

