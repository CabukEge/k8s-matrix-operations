import numpy as np


def ComputeSVD(A):
    # Konstruiere die hermitesche Matrix H
    H = np.zeros((A.shape[0] * 2, A.shape[1] * 2))
    H[:A.shape[0], A.shape[1]:] = A
    H[A.shape[0]:, :A.shape[1]] = np.conj(A.T)
    # Berechne die Eigenwerte und Eigenvektoren von H
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Sortiere die Eigenwerte und Eigenvektoren absteigend
    eigenvalues_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues_idx]
    eigenvectors = eigenvectors[:, eigenvalues_idx]
    # Konstruiere die Matrizen U, Sigma und V aus den Eigenwerten und Eigenvektoren
    U = eigenvectors[:A.shape[0], :A.shape[0]]
    V = eigenvectors[A.shape[0]:, :A.shape[0]]
    V = V.T.conj()
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        Sigma[i, i] = eigenvalues[i]
    # Normiere die Spalten von U und V
    U /= np.linalg.norm(U, axis=0)
    V /= np.linalg.norm(V, axis=0)
    return U, Sigma, V


if __name__ == "__main__":
    # Construct a random matrix for testing
    A = np.random.rand(40, 40)
    U, Sigma, V = ComputeSVD(A)

    DeltaA = U.dot(Sigma).dot(np.conj(V.T))
    print("The relative error of U * Sigma * V.H is: ", end='')
    print(np.linalg.norm(A - DeltaA) / np.linalg.norm(A))
    # Test whether U and V are unitary
    print("If U is unitary, the following number should be near zero: ", end='')
    print(np.linalg.norm(np.eye(A.shape[0]) - U.dot(np.conj(U.T))))
    print("If V is unitary, the following number should be near zero: ", end='')
    print(np.linalg.norm(np.eye(A.shape[1], A.shape[1]) - V.dot(np.conj(V.T))))