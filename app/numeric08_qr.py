import numpy as np

def QRAlgorithm(A):
    """Computes the eigenvalues of matrix A using the QR algorithm."""
    for _ in range(100):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return np.diagonal(A)

if __name__ == "__main__":
    A = np.random.rand(5, 5)
    A += A.T  # make symmetric
    eval1 = sorted(QRAlgorithm(A))
    eval2 = sorted(np.linalg.eigh(A)[0])
    print('Eigenvalues computed by QRAlgorithm():', eval1)
    print('Eigenvalues computed by numpy.linalg.eigh():', eval2)
    print('All close:', np.allclose(eval1, eval2))
