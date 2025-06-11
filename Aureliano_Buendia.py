import numpy as np
import scipy.sparse as scp
import matplotlib.pyplot as plt
from itertools import combinations

import time
import warnings
warnings.filterwarnings('ignore')

"""
Here are a lot of functions to calculate a variety of things :
-   matdens_lil calculates the density matrix of a given state in 
    scipy.sparse matrix format, for larger systems
-   matdens calculates it as a numpy array
* the basis used for these calculations is 0/1 (eigenstates of S_z)
-   sigma calculates the operator which acts as the identity on most qubits
    and as sigma_x,y or z on some qubits
-   fond_ising calculates the ground energy and ground state of the 
    Hamiltonian of the 1D transverse field Ising model

-   partialT calculates the partial transpose of a density matrix with respect
    to certain qubits in the chain
-   negat calculates the negativity of a density matrix with 
    respect to a certain subsystem
-   trace_lil calculates the partial trace over all but the specified qubits
-   matperm is used in permuto and is not very important by itself
-   permuto switches some qubits in the basis used to represent the density 
    matrix (useful in KFnorm)
-   vecvec calculates the associated matrix to a density matrix as specified 
    in eq. 2 of (https://arxiv.org/pdf/quant-ph/0205017)
-   KFnorm calculates the Ky Fan norm of a density matrix, to check the CCNR
    criterion, following the same article as the last function
    
To note that a lot of these functions are then applied to all possible subsystems
of a larger system, to check for bipartite entanglement
"""
#################################
def base(nH, état='0'):
    vect = np.zeros((2 ** nH, 1))
    vect[int(état, 2)] = 1
    return vect

def matdens_lil(etat) : 
    """
    matrice de densité de l'état csr_matrix normalisé
    
    Paramètres
    ----------
    etat : array-like déjà normalisé
        retourne la matrice densité de cet état
    """
    etat1 = scp.lil_matrix(etat)
    mat   = np.dot(etat1.T,etat1)
    return mat
def matdens(etat) : 
    """
    définition de la matrice de densité de l'état déjà normalisé
    
    etat : retourne la matrice densité de cet état (ndarray 1D)
    """
    mat    = np.outer(etat,etat)
    return mat

######################################
def partialT_old(n, rho, p=0) : # old version
    """
    transposition partielle
    Paramètres
    ----------
    n : nombre de qubits totaux (int)
    rho : matrice de densité (ndarray 2D)\n
    p : numéro du qubit à transposer (int)
    """
    d    = int(2**(n-p-1))                 # distance entre chaque petit bloc
    rhoT = rho.copy()
    
    for i in range(2**p) :            # index des blocs
        for j in range(2**p) :        # index des blocs
            rhoT[(2*i+1)*d:(2*i+2)*d, 2*j*d:(2*j+1)*d] = rho[2*i*d:(2*i+1)*d, (2*j+1)*d:(2*j+2)*d].copy()
            rhoT[2*i*d:(2*i+1)*d, (2*j+1)*d:(2*j+2)*d] = rho[(2*i+1)*d:(2*i+2)*d, 2*j*d:(2*j+1)*d].copy()
    return rhoT
def partialT(n, rho, p_list=[0]):
    """
    Tranposition partielle par rapport à certains spins

    Parameters
    ----------
    n : int
        nombre de spins.
    rho : ndarray 2**n x 2**n
        matrice de densité.
    p_list : liste de int, optional
        liste des spins à transposer. The default is [0].

    Returns
    -------
    ndarray 2**n x 2**n.

    """
    perm = np.arange(2*n)
    for p in p_list : perm[p], perm[p+n] = perm[p+n], perm[p]
    return rho.reshape([2]*(2*n)).transpose(perm).reshape((2**n,2**n))
    
def negat(n,rho, plist=[0]) :
    """
    n : nombre de qubits (int)
    rho : matrice de densité (ndarray 2D)\n
    p : numéro du/des spin à transposer (liste de int)
    """
    rho = partialT(n,rho,plist)
    vp = np.linalg.eigvalsh(rho)
    return np.sum((np.abs(vp)-vp)/2)

def trace_lil(n, etat, p_list=[0]) :
    """
    matrice de densité réduite de l'état normalisé
    
    Paramètres
    ----------
    n : int
        nombre de spins
    etat : array-like déjà normalisé
        retourne la matrice densité réduite de cet état
    p_list : list
        liste des spins qui ne sont pas tracés
    """
    tensorshape = [2]*n
    etat = etat.reshape(tensorshape)
    m = 2**len(p_list)
    
    traces = [i for i in range(n) if i not in p_list]
    rhoAAAHHH = np.tensordot(etat,etat.T,(traces,traces))
    rhoAAAHHH = rhoAAAHHH.reshape(m,m)
    return rhoAAAHHH
######################################
def matperm(d) : # old and useless
    """
    matrice qui permute deux qubits à distance d (dimension 2^(d+1))

    Parameters
    ----------
    d : int > 0
        distance entre les 2 qubits à permuter.

    Returns
    -------
    ndarray 2^(d+1) x 2^(d+1).

    """
    m = int(2**d)
    
    diag0 = np.zeros(m+m)
    diag0[     :m: 2] = 1
    diag0[m+m-1:m:-2] = 1
    
    diag1 = np.zeros(m+1)
    diag1[1::2] = 1
    
    matper = np.diag(diag0) + np.diag(diag1,m-1) + np.diag(diag1,1-m)
    return matper
def permuto_old(n, rho, p_list=[0,1]) : # old and useless
    """
    donne la matrice de densité dans une nouvelle base qui 
    inverse les spins de p_list

    Parameters
    ----------
    n : int
        nombre total de spins.
    rho : ndarray 2D
        matrice de densité.
    p_list : liste de 2 int, optional
        les positions des deux spins à permuter. The default is [0,1].

    Returns
    -------
    ndarray 2D.

    """
    d    = p_list[1]-p_list[0]
    perm = matperm(d)
    I    = np.eye(2)
    op = np.array([1])
    
    for k in range(p_list[0]) :
        op = np.kron(op, I)
    op = np.kron(op, perm)
    for k in range(n-1-p_list[1]) :
        op = np.kron(op, I)
        
    rhoB = np.matmul(op, rho)
    return rhoB

def permuto(n, rho, paires=[[0,1]]) :
    perm = np.arange(2*n)
    for p_list in paires :
        perm[p_list[0]],perm[p_list[1]]     = perm[p_list[1]],perm[p_list[0]]
        perm[p_list[0]+n],perm[p_list[1]+n] = perm[p_list[1]+n],perm[p_list[0]+n]
    return rho.reshape([2]*(2*n)).transpose(perm).reshape((2**n,2**n))

def permuto_AB(rho, dA, dB):
    n=dA+dB
    perm = np.arange(2*n)
    for p in range(min(dA,dB)):
        perm[p],perm[dA+p] = perm[dA+p],perm[p]
        perm[n+p],perm[n+dA+p] = perm[n+dA+p],perm[n+p]
    return rho.reshape([2]*(2*n)).transpose(perm).reshape((2**n,2**n))
    
def vecvec_old(rho, d_A, d_B) : # old and useless
    """
    Calcule la matrice 'tilde' pour ensuite appliquer le critère CCNR

    Parameters
    ----------
    rho : ndarray (mn x mn)
        matrice qui se fait réaligner
    d_A : int
        taille du premier système / grille de blocs = 2^(d_A) x 2^(d_A)
    d_B : int
        taille du second système / taille des blocs = 2^(d_B) x 2^(d_B)

    Returns
    -------
    rho_tilde : ndarray (2^(2*d_A) x 2^(2*d_B))

    """
    m = 2**d_A
    n = 2**d_B
    
    blocs = [np.array(rho[n*(i%m) : n*(i%m)+n , n*int(np.floor(i/m)) : n*int(np.floor(i/m))+n]) for i in range(m**2)]
    vec_blocs = [np.array([blocs[k][j%n, int(np.floor(j/n))] for j in range(n**2)]) for k in range(m**2)]
    rho_tilde = np.vstack(vec_blocs)
    
    return rho_tilde

def vecvec(rho,dA,dB) :
    m, n = 2**dA, 2**dB
    return rho.reshape((m,n,m,n)).transpose((0,2,1,3)).reshape((m*m,n*n))

def KFnorm_old(n, rho, part) : # old version
    """
    Norme de Ky Fan d'une certaine partition (KFnorm > 1 => intrication)

    Parameters
    ----------
    n : int
        nombre total de spins
    rho : ndarray 2D
        Matrice de densité (réduite) du système bipartite.
    part : liste de 2 listes
        Partition des qubits.

    Returns
    -------
    int : valeur de la norme de Ky Fan.
    """
    pmin = np.argmin([min(part[0]),min(part[1])])
    d_A, d_B = len(part[pmin]), len(part[(pmin+1)%2])
    
    for pp in part : pp = np.sort(pp)
    ppmin = part[pmin]
    
    if len(ppmin) != 1 :
        dist = 0
        pflat = np.sort(list(part[0])+list(part[1]))
        indf = np.array([np.where(pflat == p)[0][0] for p in ppmin])
        op = np.eye(2**n)
        for i in range((len(ppmin)-1)) :
            dist += indf[i+1] - indf[i] - 1
            if dist != 0 :
                op = permuto(n, op, [i+1, i+1+dist])
        rho = np.matmul(op, rho)
        
    rho_tilde = vecvec(rho, d_A, d_B)
    if d_A < d_B :
        carré = np.matmul(rho_tilde, rho_tilde.conj().T)
    else :
        carré = np.matmul(rho_tilde.conj().T, rho_tilde)
    sigmas = np.sqrt(np.round(np.linalg.eigvalsh(carré), 9))
    normeKF = np.sum(sigmas)
    #if normeKF > 1 : print(op)
    
    return normeKF

def KFnorm(n,rho,part) :
    """
    Norme de Ky Fan d'une certaine partition (KFnorm > 1 => intrication)

    Parameters
    ----------
    n : int
        nombre total de spins
    rho : ndarray 2D
        Matrice de densité (réduite) du système bipartite.
    part : liste de 2 listes
        Partition des qubits.

    Returns
    -------
    int : valeur de la norme de Ky Fan.
    """
    p0 = np.argmin([min(part[0]),min(part[1])])
    p1 = (p0+1)%2
    d_A, d_B = len(part[p0]), len(part[p1])
    
    if max(part[p0]) > min(part[p1]) :
        for pp in part : pp = np.sort(pp)
        pflat = np.sort(list(part[0])+list(part[1]))
        indf = np.array([np.where(pflat == p)[0][0] for p in part[p0]])
        paires = []
        dist = 0
        
        for i in range((len(part[p0])-1)) :
            dist += indf[i+1] - indf[i] - 1
            if dist != 0 : paires = paires + [[i+1, i+1+dist]]
        if paires != [] : rho = permuto(n,rho,paires)
    rho_tilde = vecvec(rho, d_A, d_B)
    sigmas = np.linalg.svd(rho_tilde, compute_uv = False)
    return np.sum(sigmas)
