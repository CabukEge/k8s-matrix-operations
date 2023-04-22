import numpy as np


def QR(A):
    """Given a matrix A with full column rank this function uses the classical
       Gram-Schmidt algorithm to compute a QR decomposition. It returns a tuple
       (Q, R) of np.matrix objects with Q having shape identical to A and Q*R=A."""
    # TODO: Implement this function
    c = A.shape[1]
    Q = np.matrix(np.zeros_like(A))
    R = np.matrix(np.zeros((c, c), dtype=A.dtype))
    qarray = np.matrix(np.zeros_like(A))
    for n in range(c):
        v = A[:, n]
        sigma = 0
        # Sigma bestimmen wie in Nr.2
        for i in range(n):
            qi = qarray[:, i]
            sigma = sigma + (np.multiply((v.H * qi) / (np.sum(np.square(np.absolute(qi)))), qi))
        # q_n bestimmen
        q = v - sigma
        # qn speichern
        qarray[:, n] = q
        # Spalte von Q bestimmen
        Q[:, n] = q / np.linalg.norm(q)
    # weil Q.H*Q=I gilt
    R = Q.H * A
    return Q, R


def BackSubstitution(R, y):
    """Given a square upper triangular matrix R and a vector y of same size this
       function solves R*x=y using backward substitution and returns x."""
    # TODO: Implement this function
    n = R.shape[1]
    x = np.matrix(np.zeros((n, 1), dtype=np.dtype))

    for i in range(n - 1, -1, -1):
        x[i] = y[i] / R[i, i]
        y[0:i + 1] = y[0:i + 1] - (R[0:i + 1, i] * x[i])
        print(y[0:i + 1], i)
    # funktion aus Blatt 3 -> berechnet x schneller
    # x = np.linalg.solve(R,y)
    print("my Solution:", x)
    return x


def LeastSquares(A, b):
    """Given a matrix A and a vector b this function solves the least squares
       problem of minimizing |A*x-b| and returns the optimal x."""
    q, r = QR(A);
    # r an beiden seiten links multiplizieren um r*x=r*r^-1*Q.H*b zu erhalten, wobei y=r*r^-1*Q.H*b
    y = r * np.linalg.inv(r) * Q.H * b
    # TODO: Implement this function
    # x mit der Funktion berechnen
    return BackSubstitution(r, y)

if __name__=="__main__":
    # Try the QR decomposition
    A=np.matrix([
         [1.0,1.0,1.0],
        [1.0,2.0,3.0],
        [1.0,4.0,9.0],
        [1.0,8.0,27.0]
    ])
    Q, R = QR(A)
    print("If the following numbers are nearly zero, QR seems to be working.")
    print(np.linalg.norm(Q * R-A))
    print(np.linalg.norm(Q.H * Q - np.eye(3)))
    # Try solving a least squares system
    b=np.matrix([1.0,2.0,3.0,4.0]).T
    x=LeastSquares(A, b)
    print("If the following number is nearly zero, least squares solving seems to be working.")
    print(np.linalg.norm(x-np.linalg.lstsq(A, b, rcond=-1)[0]))