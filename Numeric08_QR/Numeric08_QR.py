import numpy as np

def QRAlgorithm(A):
    """Computes the Eigenvalues of the given matrix and returns them."""
    for _ in range(100):
        Q , R = np.linalg.qr(A)
        A = np.dot(R, Q)

    return np.diagonal(A)

if __name__=="__main__":
    A = np.random.rand(5,5)
    A += A.T
    eval1 = sorted(QRAlgorithm(A))
    eval2 = sorted(sorted(np.linalg.eigh(A)[0]))
    print('Eigenvalues computed by QRAlgorithm():', eval1)
    print('Eigenvalues computed by numpy.linalg.eigh():', eval2)
    print('All close:', np.allclose(eval1, eval2))