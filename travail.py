import cvxpy as cpy
import numpy as np
import mosek
from fonctions import *
from Ising import *

'''n=3
p=3
np.random.seed(1)
C = np.random.randn(n,n)
A = []
b = []
for i in range(p):
    A.append(np.random.randn(n,n))
    b.append(np.random.randn())

X = cpy.Variable((n,n),symmetric=True)
constraints = [ X >> 0 ]
constraints += [
    cpy.trace(A[i] @ X) == b[i] for i in range(p)
]
prob = cpy.Problem(cpy.Minimize(cpy.trace(C @ X)), constraints)
prob.solve()'''
#print(prob.value)
#print(X.value)


#travail avec les polytope
# voir comment initialiser les polytope intérieur de Alice
# faire le sdp pour calculer le Ksi en maximisant t pour trouver le tau
# avec celui-ci construire le polytope de B
# A devient B et vice-versa et on recommence de faire le sdp
# lors que ksi ne change que d'une petite valeur, on arrête le code.

def RandomBlochPolytope(d: int = 2, L: int= 100):

    QPolytope = []
    for _ in range(L):
        v = np.random.randn(d) + 1j*np.random.randn(d)
        v /= np.linalg.norm(v)
        P = np.outer(v,np.conjugate(v))
        QPolytope.append(P)
    return QPolytope

def FirstRobustnessToSeparabilityBlochPolytope(rho,paires, QPolytope, solver=cpy.MOSEK,silent: bool = True):
    dimA, dimB = 2**int(len(paires[0])), 2**int(len(paires[1]))
    dim = dimA*dimB
    N = len(QPolytope)

    #variable de CVXPY
    t = cpy.Variable(nonneg=True)
    X = [cpy.Variable((dimB,dimB),hermitian=True) for _ in range(N)]

    # construit la somme entre polytope et X
    kron_blocks = []
    for k in range(N):
        Qk = QPolytope[k]
        kron_blocks.append(cpy.kron(Qk,X[k]))

    S_expr = sum(kron_blocks)
    I_dim = np.eye(rho.shape[0])

    #construire les constraintes
    constraints = []
    for k in range(N):
        constraints.append(X[k] >> 0)

    rho_mat = rho.copy()
    left_expr = t*rho_mat + ((1-t)/dim)*I_dim
    constraints.append(left_expr == S_expr)

    prob = cpy.Problem(cpy.Maximize(t),constraints)
    prob.solve(solver=solver, verbose=not silent)

    x=prob.value

    Z=[]
    for k in range(N):
        Xk_val = X[k].value
        trX = np.real(np.trace(Xk_val))
        if trX < 1e-3:
            Zk = RandomBlochPolytope(dimB,1)[0]
        else:
            Zk = Xk_val
        Z.append(Zk)

    I_B = np.eye(dimB, dtype=complex)
    Z.append((1/dimB)*I_B)

    return x, Z

def PolytopeOptimizationForWhiteRobustness(rho,paires, G, solver= cpy.MOSEK, silent: bool = True):
    dimA, dimB = 2**int(len(paires[0])), 2**int(len(paires[1]))
    dim = dimA*dimB
    N = len(G) #G est le Z
    rho_perm = perm(rho,4,paires=paires)

    #variable CVXPY
    t = cpy.Variable(nonneg=True)
    sigma_vars = [cpy.Variable((dimA,dimA),hermitian=True) for _ in range(N)]

    kron_blocks = []
    for i in range(N):
        Gi = G[i]
        kron_blocks.append(cpy.kron(Gi,sigma_vars[i]))
    S_expr = sum(kron_blocks)

    I_dim = np.eye(rho.shape[0], dtype=complex)

    constraints = []
    for i in range(N):
        constraints.append(sigma_vars[i]>>0)
        # il font une contrainte sur le PPT, mais on peut la justifié comme étant toujours vrai, car on fera des tests sur ces états-là

    left_expr = t*rho_perm + ((1-t)/dim) *I_dim
    constraints.append(left_expr == S_expr)

    prob = cpy.Problem(cpy.Maximise(t), constraints)
    prob.solve(solver=solver, verbose = not silent)

    x = prob.value

    QPolytope = []
    ResPolytope = []
    for i in range(N):
        sig_i = sigma_vars[i].value
        QPolytope.append(sig_i)

        tr_kron = np.real(np.trace(np.kron(G[i],sig_i)))
        if tr_kron> 0.01:
            u = sig_i.copy()
            u /= np.trace(u)
            ResPolytope.append(u)

        if np.real(np.trace(sig_i)) < 0.001:
            QPolytope[i] = RandomBlochPolytope(dimA,1)[0]
        if np.isclose(np.trace(QPolytope[i]),0.0):
            print('Warning: some trace is zero')

    return QPolytope, ResPolytope


def RobustnessToSeparabilityByBlochPolytope(
    rho,
    QPolytope,
    paires=[[0,1],[2,3]],
    num_iter: int =1,
    solver = cpy.MOSEK,
    convergence_recognition: bool = True,
    convergence_accuracy: float = 1e-4,
    silent: bool = True
):

    x=0.0
    c=0
    d=1
    ResPolytope = []
    if num_iter ==1:
        ResPolytope= QPolytope.copy()

    while c<num_iter:
        U_x, Z = FirstRobustnessToSeparabilityBlochPolytope(rho,paires,QPolytope,solver,silent)
        if c > 0 and convergence_recognition:
            if abs(U_x-x <= convergence_accuracy):
                break
        x = U_x
        if num_iter > 1 and c < (num_iter-1):
            QPolytope,ResPolytope = PolytopeOptimizationForWhiteRobustness(rho,paires, Z, solver, silent)
        d+=1
        c+=1

    return x,Z,QPolytope,ResPolytope,d



polytechniquemontreal = RandomBlochPolytope(2) # le polytope créer doit avoir le même nombre d'état que dA
testis = RobustnessToSeparabilityByBlochPolytope(test012,polytechniquemontreal,[[0],[1,2,3]]) #inclure truc pour les paires

#la fonctionne est bonne pour des 2vs2, mais voir pour des 1vs3