import numpy as np

def QR(A):
    """Computes a QR decomposition using Gram-Schmidt."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return np.matrix(Q), np.matrix(R)

def BackSubstitution(R, y):
    """Solves R*x = y for an upper triangular R."""
    n = R.shape[0]
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = y[i] / R[i, i]
        y[:i] -= R[:i, i].reshape(-1, 1) * x[i]
    return np.matrix(x)

def LeastSquares(A, b):
    """Solves least squares using QR decomposition."""
    print("DEBUG - LeastSquares Called")  

    Q, R = QR(A)  

    print("DEBUG - Q shape:", Q.shape)  
    print("DEBUG - R shape:", R.shape)  

    y = Q.T @ b
    x = BackSubstitution(R, y)

    print("DEBUG - Least Squares Solution Shape:", x.shape)  

    return x


if __name__=="__main__":
    A = np.matrix([
         [1.0, 1.0, 1.0],
         [1.0, 2.0, 3.0],
         [1.0, 4.0, 9.0],
         [1.0, 8.0, 27.0]
    ])
    Q, R = QR(A)
    print("QR decomposition check, norm:", np.linalg.norm(Q @ R - A))
    b = np.matrix([1.0, 2.0, 3.0, 4.0]).T
    x = LeastSquares(A, b)
    print("Least squares solution:", x)
