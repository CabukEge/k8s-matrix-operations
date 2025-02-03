import numpy as np
# The following IPython imports are not required in the API but are kept for completeness.
from IPython.display import Markdown, display

def print_md(md):
    display(Markdown(md))

def array_to_md(A, precision=5):
    A = np.round(A, precision)
    md = "$\\begin{pmatrix}"
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            entry = str(A[i, j])
            md += entry + "&"
        md = md[:-1] + '\\\\'
    md = md[:-2]
    md += "\\end{pmatrix}$"
    return md

def LUP(A):
    """Computes an LU decomposition with pivoting such that LU = P*A."""
    L = np.eye(A.shape[0])
    U = np.copy(A)
    P = np.eye(A.shape[0])
    m = A.shape[0]
    for k in range(m - 1):
        i = np.argmax(abs(U[k:, k])) + k
        U[[k, i], k:m] = U[[i, k], k:m]
        L[[k, i], :k] = L[[i, k], :k]
        P[[k, i], :] = P[[i, k], :]
        for j in range(k + 1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:m] -= L[j, k] * U[k, k:m]
    return L, U, P

def ForwardSubstitution(L, b):
    n = L.shape[0]
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = b[i] / L[i, i]
        if i + 1 < n:  # Only perform this if there's a next row
            b[i+1:n] -= L[i+1:n, i] * x[i]
    return x


def BackSubstitution(U, b):
    n = U.shape[0]
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / U[i, i]
        b[:i] -= U[:i, i].reshape(-1, 1) * x[i]
    return x

def SolveLinearSystemLUP(A, b):
    print("DEBUG - SolveLinearSystemLUP: Matrix A shape:", A.shape)
    print("DEBUG - SolveLinearSystemLUP: Vector b shape:", b.shape)

    L, U, P = LUP(A)

    print("DEBUG - LUP Decomposition: L shape:", L.shape)
    print("DEBUG - LUP Decomposition: U shape:", U.shape)
    print("DEBUG - LUP Decomposition: P shape:", P.shape)

    y = ForwardSubstitution(L, P @ b)
    print("DEBUG - After Forward Substitution, y shape:", y.shape)

    x = BackSubstitution(U, y)
    print("DEBUG - Solution x shape:", x.shape)

    return x


def LeastSquares(A, b):
    """
    Solves the least squares problem min|A*x - b| using QR decomposition.
    """
    Q, R = QR(A)
    y = Q.T @ b
    x = BackSubstitution(R, y)
    return x


if __name__ == "__main__":
    A = np.array([[1, 2, 6],
                  [4, 8, -1],
                  [2, 3, 5]], dtype=np.double)
    b = np.array([[1], [2], [3]], dtype=np.double)
    L, U, P = LUP(A)
    print("Zero (LUP sanity check):", np.linalg.norm(L @ U - P @ A))
    print("Zero (SolveLinearSystemLUP sanity check):", np.linalg.norm(A @ SolveLinearSystemLUP(A, b) - b))
    A_test = np.random.rand(6, 4)
    b_test = np.random.rand(6)
    print("Zero (LeastSquares sanity check):", np.linalg.norm(LeastSquares(A_test, b_test).flat - np.linalg.lstsq(A_test, b_test, rcond=None)[0]))
