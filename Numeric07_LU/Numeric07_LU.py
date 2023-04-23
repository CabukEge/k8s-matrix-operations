import numpy as np
from IPython.display import Markdown, display
import scipy as sp

def print_md(md):
    display(Markdown(md))
def array_to_md(A, precision=5):
    A = np.round(A, precision)
    md = "$\\begin{pmatrix}"
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            entry = str(A[i,j])
            md += entry + "&"
        md = md[:-1] + '\\\\' # strip last & and add \\
    md = md[:-2] # strip last \\
    md += "\\end{pmatrix}$"
    return md


def LUP(A):
    """Computes and returns an LU decomposition with pivoting. The return value
       is a tuple (L,U,P) and fulfills LU=PA (* is matrix-multiplication)."""
    L = np.eye(A.shape[0])  # TODO: Implement the function
    U = np.copy(A)  # TODO: Implement the function
    P = np.eye(A.shape[0])  # TODO: Implement the function
    m = A.shape[0]
    for k in range(m - 1):
        i = np.argmax(abs(U).T[k, k:]) + k
        # vertausche Zeilen wie in 6.2 angegeben
        U[[k, i], k:m] = U[[i, k], k:m]
        L[[k, i], 0:k] = L[[i, k], 0:k]
        P[[k, i], :] = P[[i, k], :]
        for j in range(k + 1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:m] = U[j, k:m] - (L[j, k] * U[k, k:m])

    return L, U, P


def ForwardSubstitution(L, b):
    """Solves the linear system of equations L*x=b assuming that L is a left lower
       triangular matrix. It returns x as column vector."""

    n = L.shape[1]
    x = np.zeros((L.shape[1], 1))
    # Algorithmus 4.1
    for i in range(n):
        x[i] = b[i] / L[i, i]
        # b transponieren, da L*x horizontal ist
        # -> danach zurueck transponieren, da x = vertikaler Vektor ist
        b[i + 1:n] = (b[i + 1:n].T - (L[i + 1:n, i] * x[i])).T
    return x  # TODO: Implement the function


def BackSubstitution(U, b):
    """Solves the linear system of equations U*x=b assuming that U is a right upper
       triangular matrix. It returns x as column vector."""
    n = U.shape[1]
    x = np.zeros((U.shape[1], 1))
    # Algorithmus 4.2
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / U[i, i]
        # b transponieren, da L*x horizontal ist
        # -> danach zurueck transponieren, da x = vertikaler Vektor sein soll
        b[0:i] = (b[0:i].T - (U[0:i, i] * x[i])).T
    return x  # TODO: Implement the function


def SolveLinearSystemLUP(A, b):
    """Given a square array A and a matching vector b this function solves the
       linear system of equations A*x=b using a pivoted LU decomposition and returns
       x."""
    L, U, P = LUP(A)
    # 6.1 mit  Pivotisierung
    y = ForwardSubstitution(L, P @ b)
    x = BackSubstitution(U, y)
    return x  # TODO: Implement the function


def LeastSquares(A, b):
    """Given a matrix A and a vector b this function solves the least squares
       problem of minimizing |A*x-b| and returns the optimal x."""
    # 2.1
    x = SolveLinearSystemLUP(A.transpose() @ A, A.transpose() @ b)
    return x  # TODO: Implement the function

# A test matrix where LU fails but LUP works fine
A=np.array([[1,2, 6],
            [4,8,-1],
            [2,3, 5]],dtype=np.double)
b=np.array([[1],[2],[3]],dtype=np.double)
# Test the LUP-decomposition
L,U,P=LUP(A)
print_md("$A=$" + array_to_md(A))
print_md("$L=$" + array_to_md(L))
print_md("$U=$" + array_to_md(U))
print_md("$P=$" + array_to_md(P))
print_md("$LU=$" + array_to_md(L@U) + "$\\stackrel{?}{=}$" + array_to_md(P@A) + "=PA")
print("Zero (LUP sanity check): "+str(np.linalg.norm(np.dot(L,U) - np.dot(P,A))))
# Test the method for solving a system of linear equations
print("Zero (SolveLinearSystemLUP sanity check): " + str(np.linalg.norm(np.dot(A, SolveLinearSystemLUP(A, b)) - b)))
# Test the method for solving linear least squares
A=np.random.rand(6,4)
b=np.random.rand(6)
print("Zero (LeastSquares sanity check): " + str(np.linalg.norm(LeastSquares(A, b).flat - np.linalg.lstsq(A, b, rcond=None)[0])))

