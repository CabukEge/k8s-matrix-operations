import numpy as np

def ComputeSVD(A):
    """
    Computes the SVD of A using a Hermitian augmentation.
    Returns (U, Sigma, V) where U and V are unitary.
    """
    H = np.zeros((A.shape[0]*2, A.shape[1]*2), dtype=complex)
    H[:A.shape[0], A.shape[1]:] = A
    H[A.shape[0]:, :A.shape[1]] = np.conj(A.T)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    U = eigenvectors[:A.shape[0], :A.shape[0]]
    V = eigenvectors[A.shape[0]:, :A.shape[0]]
    V = V.T.conj()
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    for i in range(min(A.shape[0], A.shape[1])):
        Sigma[i, i] = abs(eigenvalues[i])
    U = U / np.linalg.norm(U, axis=0)
    V = V / np.linalg.norm(V, axis=0)
    return U.real, Sigma.real, V.real

if __name__ == "__main__":
    A = np.random.rand(40, 40)
    U, Sigma, V = ComputeSVD(A)
    DeltaA = U @ Sigma @ V.T
    print("Relative error:", np.linalg.norm(A - DeltaA) / np.linalg.norm(A))
    print("Unitary check U:", np.linalg.norm(np.eye(A.shape[0]) - U @ U.T))
    print("Unitary check V:", np.linalg.norm(np.eye(A.shape[1]) - V @ V.T))
