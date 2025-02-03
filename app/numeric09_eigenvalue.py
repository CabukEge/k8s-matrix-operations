import numpy as np

def PowerIteration(A, v):
    """
    Approximates the largest eigenvalue using power iteration.
    Returns a list of successive eigenvalue estimates.
    """
    vlist = [v]
    lambdalist = []
    for k in range(1, 101):
        w = A @ vlist[k-1]
        v_next = w / np.linalg.norm(w)
        vlist.append(v_next)
        lambdalist.append(float(v_next.conj().T @ A @ v_next))
    return lambdalist

def RayleighQuotientIteration(A, v):
    """
    Uses Rayleigh quotient iteration to approximate an eigenvalue near the
    initial vector v. Returns successive approximations.
    """
    vlist = [v]
    lam = float(v.conj().T @ (A @ v))
    lambdalist = [lam]
    for k in range(1, 101):
        try:
            w = np.linalg.solve(A - lambdalist[k-1] * np.eye(A.shape[0]), vlist[k-1])
        except np.linalg.LinAlgError:
            break
        v_next = w / np.linalg.norm(w)
        vlist.append(v_next)
        lam = float(v_next.conj().T @ A @ v_next)
        lambdalist.append(lam)
    return lambdalist

if __name__ == "__main__":
    print("Module numeric09_eigenvalue is imported successfully.")
