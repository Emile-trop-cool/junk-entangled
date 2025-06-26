#%% # modules & cie
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations
from icecream import ic
import pickle
from time import time
#from Aureliano_Buendia import *
#from Hspins import *

def save_obj(obj, filename="data"):
    try:
        with open(filename+".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
def load_obj(filename="data"):
    try:
        with open(filename+".pickle", "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

####################################################################################################
#%% # fonctions polytopialtes
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

def SDP_polytope(rho, d_A, d_B, polytope_A, n_spin=2,yap=False):
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
    if yap : print(problem.status)

    tau = []
    for i in range(N):
        x_val = Y[i].value
        if np.real(np.trace(x_val)) < 1e-3:
            x_val = random_polytope(d_B, 1, n_spin)[0]
        tau.append(x_val)
    tau.append((1 / n_spin**d_B) * np.eye(n_spin**d_B, dtype=complex))

    return problem.value, tau

def test_polytope(rho, d_A, d_B, polytope, num_iter=1, convergence_recognition=True, convergence_accuracy=1e-4, n_spin=2,yap=False):
    x = 0.0
    c = 0
    rho1 = rho
    rho2 = permuto_AB(rho, d_A, d_B, n_spin)
    rholist = [rho1, rho2]
    while c < num_iter:
        time11 = time()
        
        rhoC = rholist[c % 2]
        t_val, polytope = SDP_polytope(rhoC, d_A, d_B, polytope, n_spin, yap)
        if c > 0 and convergence_recognition:
            if abs(t_val - x) <= convergence_accuracy:
                break
        x = t_val.copy()
        
        time22 = time()
        if yap : print(f"après {c+1} itérations: {x}, temps : {time22-time11}")

        d_A, d_B = d_B, d_A
        c += 1

    return t_val, polytope, c


def lowerbound(n, rho, precision=4) : # valeur de t à laquelle la négativité apparaît
    tlist = np.linspace(0,1,11)
    for j in range(1,precision+1) :
        for t in tlist :
            rhot = t*rho + ((1-t)/2**n)*np.eye(2**n)
            neg = negat(n,rhot,range(int(n/2)))
            if neg != 0 :break
            tmax = t
        if tmax == 1 : break
        tlist = np.linspace(tmax,tmax+10**(-j),10,False)
    if tmax != 1 : tmax += 10**(-precision)
    return tmax

####################################################################################################
#%% # états aléatoires

m   = 3   # l'état sera une bipartition de m vs m qubits
dec = 6   # nombre de décimales de précision
taillepoly = 250
niter = 20

ic(dec,taillepoly,niter)

for kk in range(200):
    etatpur  = np.random.randn(16**m) + 1j*np.random.randn(16**m)
    etatpur /= np.linalg.norm(etatpur)
    rhored   = trace_lil(4*m, etatpur, range(2*m))
    lb = np.round(lowerbound(2*m, rhored, dec+1),dec)
    pt = np.round(test_polytope(rhored, m, m, random_polytope(m,taillepoly),niter,convergence_accuracy=10**(-dec))[0],dec)
    if np.round(lb-pt,dec-1) != 0 :
        print('!!!!!!!!!!!!!!!!!!!!!!')
        save_obj(rhored,f"matrice{kk}")
    print(f"t max pour être PPT : {lb}, t max du polytope : {pt}"
          +f"\ndifférence (PPT-poly) : {np.round(lb-pt,dec+1)}")


##### tests débuggage

etatpur  = np.random.randn(256) + 1j*np.random.randn(256)
etatpur /= np.linalg.norm(etatpur)
rhomal   = trace_lil(8, etatpur, range(4))
#rhomal = load_obj('matrice0')
#ic(rhomal)

lblb = lowerbound(4, rhomal,16)
ic(lblb)
ptptab6 = []
ptptab7 = []
for i in range(7) :
    time1 = time()
    npol = 50+25*i
    
    ptpt6,polpol,d6 = test_polytope(rhomal, 2, 2, random_polytope(2,npol),40,True,1e-6)
    ptptab6.append(ptpt6)
    time2=time()
    
    ptpt7,polpol,d7 = test_polytope(rhomal, 2, 2, random_polytope(2,npol),40,True,1e-7)
    ptptab7.append(ptpt7)
    time3=time()
    
    ic(npol, d6, d7)
    ic(ptpt6, ptpt7)
    print(f'temps pris 6 : {time2-time1}')
    print(f'temps pris 7 : {time3-time2}\n')



# Idéal : 1e-6, taille=75 !!!!!!!!!!!!!!!!!!

####################################################################################################
#%% # Horodecki

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

####################################################################################################
#%% # GHZ+W

dA, dB = 1,3
niter = 20

GHZ = (base(4,'1111')+base(4,'0000'))/np.sqrt(2)
W   = (base(4,'0001')+base(4,'1000')+base(4,'0100')+base(4,'0010'))/2
rho_bench = matdens(W)
X, poly, cd = test_polytope(rho_bench, dA, dB, random_polytope(dA, 200), niter, True, 1e-6,2,1)

####################################################################################################
#%% # ising / XX

n = 14
h = 1
Jx = 1
Jy = 1

dA = 2
dB = 2

niter = 30
taillepoly = 75
if0 = 1

NPT    = 0
PPT    = 0
CCNR   = 0
POLSEP = 0
POLINT = 0

etat_fond = np.array(full_low_XX(n, 1, Jx, Jy, h)[1])

ic(n,h,niter,taillepoly)

Qlist = combinations(range(n), dA+dB)
#Qlist = [ (1,3,5,i,i+2,i+4) for i in range(6,11) ]

pairlist = [[]]

for comb in Qlist:
    if ((if0==False) or 0 in comb) :
        rholil = trace_lil(n, etat_fond, comb)
        #for i in range(1,4) :
        #    if i!=1 : rholil = permuto(4, rholil, paires=[[1,i]])
        #rholil = permuto(6, rholil, paires=[[2,3]])
        negativite = np.round(float(negat(dA+dB, rholil, range(dA))),10)
        ic(comb,negativite,np.round(rholil.sum(),10))
        if negativite == 0:
            PPT += 1
            KyFan = KFnorm(dA+dB, rholil, [range(dA),range(dA,dA+dB)])
            if KyFan > 1 : 
                CCNR += 1
                print(f"{comb}, - cross-norm={KyFan}, intrication liée confirmée !!!!!!")
            else :
                X, poly, cd = test_polytope(rholil, 3, 3, random_polytope(3,taillepoly), niter,True,1e-4,yap=True)
                if np.round(X,4)==1 : 
                    POLSEP += 1
                    sepbound = "séparable"
                else : 
                    POLINT += 1
                    sepbound = "intrication liée possible"
                print(f"{comb}, - chi={np.round(X,6)} ({cd} itérations) ------- "+sepbound)
        else : NPT += 1

ic(n,h,Jx,Jy,niter,taillepoly)
ic(NPT,PPT,CCNR,POLSEP,POLINT)

#%% # ising en profondeur

n = 18
h = 1
etat_fond = fond_ising(n, h)[1]

comb = (0,1,3,7,8,11)
i = 1
niter = 50
taillepoly = 500

rholil = trace_lil(n, etat_fond, comb)
#if i!=1 : rholil = permuto(4, rholil, paires=[[1,i]])
X, poly, cd = test_polytope(rholil, 3, 3, random_polytope(3,taillepoly), niter,True,1e-6,yap=True)



#%% # ising apparition de PPT

nmin=10
limite=0

def test_magique(n,h):
    
    Qlist = combinations(range(n), 4)
    etat_fond = fond_ising(n, h)[1]
    NPT = 0
    PPT = 0
    
    for comb in Qlist:
        if 0 in comb :
            rholil = trace_lil(n, etat_fond, comb)
            for i in range(1,4) :
                negativite = np.round(float(negat(4, rholil, [0,i])),10)
                if negativite == 0: PPT += 1
                else : NPT += 1
    return PPT, NPT

n_tab = range(nmin,21)
h_tab = np.zeros(len(n_tab))

for k in range(len(n_tab)) :
    n = n_tab[k]
    ppt = 0
    h0 = 0
    for i in range(1,6) :
        h = h0
        while True :
            #print(f"{h}, ",end='')
            ppt, npt = test_magique(n,h)
            ic(n,h,ppt,npt)
            if ppt != 0 and h > limite : break
            h0 = h
            h += 10**(-i)
            h = np.round(h,i+1)
            if h >= 10 : break
        if h >= 10 : 
            h = None
            break
    h_tab[k] = h0
    print(f"\n{n} qubits, complété\n")
        
#%% scan PPT

#n = 15
hmin = 0
hmax = 5.0000001
precision = 0.02
    
for n in range(10,16):
    htab = np.arange(hmin,hmax,precision)
    PPTab= np.zeros(len(htab))
    
    for k in range(len(htab)) :
        PPTab[k] = test_magique(n, htab[k])[0]
    print(f"{n} : valeur asymptotique du nombre de PPT : {PPTab[-1]}")
    
    plt.plot(htab,PPTab, label='{n}', linewidth=6)
