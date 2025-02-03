import numpy as np

def ComputeSVD(A):
    """
    Computes an SVD decomposition using the eigenvalue decomposition of A.T @ A.
    Returns (U, Sigma, V) as NumPy matrices.
    """
    m, n = A.shape
    B = A.T @ A
    e, ev = np.linalg.eig(B)
    V = np.matrix(ev)
    singVal = np.sqrt(e)
    Sigma = np.diag(singVal)
    U = np.zeros((m, len(singVal)))
    for i in range(len(ev)):
        if singVal[i] != 0:
            U[:, i] = (1 / singVal[i]) * A @ ev[:, i]
        else:
            U[:, i] = A @ ev[:, i]
    U = np.matrix(U)
    return U, Sigma, V

def PseudoInverse(A):
    """
    Computes the pseudoinverse of A using its SVD.
    """
    U, Sigma, V = ComputeSVD(A)
    SigmaInv = np.linalg.pinv(Sigma)
    return V @ SigmaInv @ U.T

def LinearSolve(A, b):
    """
    Solves A*x = b via the pseudoinverse.
    """
    APsInv = PseudoInverse(A)
    return APsInv @ b

if __name__ == "__main__":
    A = np.matrix([
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
    b = np.matrix([1.0, 2.0, 3.0, 4.0]).T
    x = LinearSolve(A, b)
    print("Solution x:")
    print(x)
