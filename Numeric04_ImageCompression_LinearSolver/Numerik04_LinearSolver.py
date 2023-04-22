import numpy as np
from numpy.linalg import eigh, eig
from math import sqrt


def ComputeSVD(A):
    """
       Given a matrix A use the eigen value decomposition to compute a SVD
       decomposition. It returns a tuple (U,Sigma,V) of np.matrix objects.
    """
    m, n = A.shape
    #B generieren
    B = np.matmul(A.transpose(), A)
    #Eigenwerte und Eigenvektoren generieren
    e ,ev = np.linalg.eig(B)
    print(B)
    print("________________")
    print(e)
    print("________________")
    print(ev)
    print("________________")

    #V generieren
    V = np.matrix(ev)

    #Sigma generieren

    singVal = np.sqrt(e)
    sig = np.diag(singVal)


    print("sig:")
    print(sig)
    print("________________")


    #U generieren
    U = np.zeros((m,0))
    for i in range(len(ev)):
        U = np.concatenate((U,(1 / singVal[i]) * A * ev[:,i]), axis=1)
    return U, sig, V# TODO: implement the function


def PseudoInverse(A):
    """
        Given a matrix A use the SVD of A to compute the pseudo inverse.
        It returns the pseudo inverse as a np.matrix object.
    """
    U, Sig, V = ComputeSVD(A)
    SigInv = np.linalg.inv(Sig)
    Utrans = np.transpose(U)
    APsInv = V*SigInv*Utrans
    return APsInv # TODO: Implement the function


def LinearSolve(A, b):
    """
        Given a matrix A and a vector b this function solves the linear
        equations A*x=b by solving the least squares problem of minimizing
        |A*x-b| and returns the optimal x.
    """
    APsInv = PseudoInverse(A)
    return APsInv*b # TODO: Implement the function
    
if __name__ == "__main__":
    # Try the SVD decomposition
    A = np.matrix([ #1^i 2^i 3^i
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.0, 4.0, 9.0],
        [1.0, 8.0, 27.0]
    ])

    U, Sigma, V = ComputeSVD(A)
    print("U:")
    print(U)
    print("Sigma:")
    print(Sigma)
    print("V:")
    print(V)
    print("If the following numbers are nearly zero, SVD seems to be working.")
    print(np.linalg.norm(U * Sigma * V.H - A))
    print(np.linalg.norm(U.H * U - np.eye(3)))
    print(np.linalg.norm(V.H * V - np.eye(3)))
    # Try solving a least squares system
    b = np.matrix([1.0, 2.0, 3.0, 4.0]).T
    x = LinearSolve(A, b)
    print("If the following number is nearly zero, "
          "linear solving seems to be working.")
    print(np.linalg.norm(x - np.linalg.lstsq(A, b, rcond=None)[0]))