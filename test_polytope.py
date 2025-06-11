import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations
from icecream import ic
#from Aureliano_Buendia import *
#from Hspins import *

def random_polytope(d=1, L=100, n_spin=2):
    """
    Generates a random Bloch polytope of rank-1 projectors (density matrices).

    Args:
        d (int): Number of qudits (default 1).
        L (int): Number of vertices in the polytope.
        n_spin (int): dimension of the 1-ptcle space

    Returns:
        List[np.ndarray]: A list of d x d Hermitian rank-1 projectors.
    """
    q_polytope = []
    for _ in range(L):
        V = np.random.randn(n_spin**d) + 1j * np.random.randn(n_spin**d)
        V /= np.linalg.norm(V)
        q_polytope.append(np.outer(V, V.conj()))
    return q_polytope

def permuto_AB(rho, dA, dB, n_spin=2):
    n = dA+dB
    perm = np.arange(2*n)
    for p in range(min(dA, dB)):
        perm[p], perm[dA+p] = perm[dA+p], perm[p]
        perm[n+p], perm[n+dA+p] = perm[n+dA+p], perm[n+p]
    return rho.reshape([n_spin]*(2*n)).transpose(perm).reshape((n_spin**n, n_spin**n))

def SDP_polytope(rho, d_A, d_B, polytope_A, n_spin=2):
    N = len(polytope_A)
    t = cp.Variable(nonneg=True)
    Y = [cp.Variable((n_spin**d_B, n_spin**d_B), hermitian=True) for k in range(N)]
    S_expr = sum([cp.kron(polytope_A[k], Y[k]) for k in range(N)])
    d_total = d_A + d_B
    identity = np.eye(n_spin**d_total)
    target = t * rho + ((1 - t) / n_spin**d_total) * identity
    
    constraints = [S_expr == target]+[Y[i] >> 0 for i in range(N)]
    problem = cp.Problem(cp.Maximize(t), constraints)
    problem.solve(solver=cp.MOSEK)#, verbose=True)
    # print(problem.status)

    tau = []
    for i in range(N):
        x_val = Y[i].value
        if np.real(np.trace(x_val)) < 1e-3:
            x_val = random_polytope(d_B, 1, n_spin)[0]
        tau.append(x_val)
    tau.append((1 / n_spin**d_B) * np.eye(n_spin**d_B, dtype=complex))

    return problem.value, tau

def test_polytope(rho, d_A, d_B, polytope, num_iter=1, convergence_recognition=True, convergence_accuracy=1e-4, n_spin=2):
    x = 0.0
    c = 0
    rho1 = rho
    rho2 = permuto_AB(rho, d_A, d_B, n_spin)
    rholist = [rho1, rho2]
    while c < num_iter:
        rhoC = rholist[c % 2]
        t_val, polytope = SDP_polytope(rhoC, d_A, d_B, polytope, n_spin)
        if c > 0 and convergence_recognition:
            if abs(t_val - x) <= convergence_accuracy:
                break
        x = t_val.copy()
        #print(f"après {c+1} itérations: {x}")

        d_A, d_B = d_B, d_A
        c += 1

    return t_val, polytope, c+1

def lowerbound(n, rho, precision=4) :
    tlist = np.linspace(0,1,11)
    for j in range(1,precision+1) :
        for t in tlist :
            rhot = t*rho + ((1-t)/2**n)*np.eye(2**n)
            neg = negat(n,rhot,range(int(n/2)))
            if neg != 0 :break
            tmax = t
        if tmax == 1 : break
        tlist = np.linspace(tmax,tmax+10**(-j),10,False)
    return tmax

###################################################
''' états aléatoires
m=2
for _ in range(100):
    rhoasym = np.random.randn(4**m,4**m)
    rhosym  = (rhoasym+rhoasym.T)/2
    lb = np.round(lowerbound(2*m, rhosym, 6),6)
    pt = np.round(test_polytope(rhosym, m, m, random_polytope(2,200),20,convergence_accuracy=1e-6)[0],6)
    print(f"t max pour être PPT : {lb}, t max du polytope : {pt}"
          +f"\ndifférence (PPT-poly) : {np.round(lb-pt,7)}")
'''
###################################################
''' Horodecki
def Horo33(a) :
    aa  = (1+a)/2
    aaa = np.sqrt(1-a**2)/2
    c   = 1/(8*a+1)
    return c*np.array([[a,0,0,0,a,0,0,0,a],
                       [0,a,0,0,0,0,0,0,0],
                       [0,0,a,0,0,0,0,0,0],
                       [0,0,0,a,0,0,0,0,0],
                       [a,0,0,0,a,0,0,0,a],
                       [0,0,0,0,0,a,0,0,0],
                       [0,0,0,0,0,0,aa,0,aaa],
                       [0,0,0,0,0,0,0,a,0],
                       [a,0,0,0,a,0,aaa,0,aa]])

lista = np.linspace(0,1,11)
listchi = np.zeros(11)
for i in range(6):
    listchi[i] = test_polytope(Horo33(lista[i]), 1, 1, random_polytope(n_spin=3),10,n_spin=3)[0]
plt.plot(lista,listchi)

def Horo24(a) : # b=a
    aa  = (1+a)/2
    aaa = np.sqrt(1-a**2)/2
    c   = 1/(7*a+1)
    return c*np.array([[a,0,0,0,0,a,0,0],
                       [0,a,0,0,0,0,a,0],
                       [0,0,a,0,0,0,0,a],
                       [0,0,0,a,0,0,0,0],
                       [0,0,0,0,aa,0,0,aaa],
                       [a,0,0,0,0,a,0,0],
                       [0,a,0,0,0,0,a,0],
                       [0,0,a,0,aaa,0,0,aa]])

lista = np.linspace(0,1,11)
listchi = np.zeros(11)
for i in range(11) :
    listchi[i] = test_polytope(Horo24(lista[i]), 1, 2, random_polytope(),10,n_spin=2)[0]
plt.plot(lista,listchi)
plt.show()
'''
###################################################
''' ising ?
n, h = 15, 0.5
dA = 1
dB = 3
niter = 20

etat_fond = fond_ising(n, h)[1]

Qlist = combinations(range(n), dA+dB)
for comb in Qlist:
    rholil = trace_lil(n, etat_fond, comb)
    X, poly, cd = test_polytope(rholil, dA, dB, random_polytope(dA), niter)
    neg = negat(dA+dB, rholil)
    
    concorde = True
    if np.round(neg, 8) != 0. and np.round(X, 3) >= 1.:
        concorde = False

    print(comb, ' : négativité ', neg, '; visibilité ',
          X, '\n', concorde)
'''
###################################################
''' GHZ+W
GHZ = (base(4,'1111')+base(4,'0000'))/np.sqrt(2)
W   = (base(4,'0001')+base(4,'1000')+base(4,'0100')+base(4,'0010'))/2
rho_bench = matdens(W)
X, poly, cd = test_polytope(rho_bench, dA, dB, random_polytope(dA, 200), niter, True, 1e-6)
'''
'''the cake is a lie'''
