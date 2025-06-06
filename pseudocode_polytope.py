import numpy as np
import cvxpy as cp


def random_polytope(d=1, L=100):
    """
    Generates a random Bloch polytope of rank-1 projectors (density matrices).
    
    Args:
        d (int): Dimension of the Hilbert space (default 2).
        L (int): Number of vertices in the polytope.
    
    Returns:
        List[np.ndarray]: A list of d x d Hermitian rank-1 projectors.
    """
    q_polytope = []
    for _ in range(L):
        V = np.random.randn(2**d) + 1j * np.random.randn(2**d)
        V /= np.linalg.norm(V)
        q_polytope.append(np.outer(V, V.conj()))
    return q_polytope

def permuto_AB(rho, dA, dB):
    n=dA+dB
    perm = np.arange(2*n)
    for p in range(min(dA,dB)):
        perm[p],perm[dA+p] = perm[dA+p],perm[p]
        perm[n+p],perm[n+dA+p] = perm[n+dA+p],perm[n+p]
    return rho.reshape([2]*(2*n)).transpose(perm).reshape((2**n,2**n))

def SDP_polytope(rho, d_A, d_B, polytope_A):
    N   = len(polytope_A)
    t = cp.Variable(nonneg=True)
    Y = [cp.Variable((2**d_B, 2**d_B), hermitian=True) for _ in range(N)]
    
    S_expr = sum([cp.kron(polytope_A[k], Y[k]) for k in range(N)])

    d_total = d_A + d_B
    identity = np.eye(2**d_total)
    target = t * rho + ((1 - t) / 2**d_total) * identity

    constraints = [S_expr == target]+[Y[i]>>0 for i in range(N)]
    problem = cp.Problem(cp.Maximize(t), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)
    print(problem.status)
    
    tau = []
    for i in range(N):
        x_val = Y[i].value
        if np.real(np.trace(x_val)) < 1e-3:
            x_val = random_polytope(d_B, 1)[0]
        tau.append(x_val)
    tau.append((1 / 2**d_B) * np.eye(2**d_B, dtype=complex))
    
    return t.value, tau

def test_polytope(rho, d_A, d_B, polytope, num_iter=1, convergence_recognition=True, convergence_accuracy=1e-4):
    x = 0.0
    c = 0
    rho1 = rho
    rho2 = permuto_AB(rho, d_A, d_B)
    rholist = [rho1, rho2]
    
    while c < num_iter:
        rhoC = rholist[c%2]
        t_val, polytope = SDP_polytope(rhoC, d_A, d_B, polytope)
        if c > 0 and convergence_recognition:
            if abs(t_val - x) <= convergence_accuracy:
                break
        x = t_val
        print(f"after {c+1} iterations: {x}")
        
        d_A,d_B = d_B,d_A
        c += 1
    
    return x, polytope, c+1

###################################################
n, h = 14, 0.5
comb = (1,3,4,5)

etat_fond = fond_ising(n,h)[1]
rholil = trace_lil(n, etat_fond, comb)

X, poly, cd = test_polytope(rholil, 1, 3, random_polytope(1), 2)
