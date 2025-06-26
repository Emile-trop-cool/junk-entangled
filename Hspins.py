# Liste de fonctions qui donnent les états fondamentaux d'hamiltoniens
# stoquastiques, et de spectres presque complets parfois.

import numpy as np
import scipy.sparse as scp
#from Aureliano_Buendia import *

#%% # fonctions de base :
def sigma(n=1, i_list=[0], xyz='z', cste=1) :
    """
    Opérateur spin dans une direction cardinale pour un spin dans une chaîne

    Parameters
    ----------
    n : int
        nombre de spins dans la chaîne. The default is 1.
    i : liste de int
        opérateur(s) de(s) quel(s) spin(s) entre 0 et n-1. The default is 0.
    xyz : str
        quelle(s) direction(s) entre x, y ou z. The default is 'z'.
    cste : complex ou float ou int
        multiplie la matrice par cette constante

    Returns
    -------
    ndarray (matrice 2^n par 2^n de l'opérateur).

    """
    X = scp.csr_matrix([[0,1],[1,0]])
    Y = scp.csr_matrix([[0,-1j],[1j,0]])
    Z = scp.csr_matrix([[1,0],[0,-1]])
    I = scp.eye(2,format='csr')
    
    op_list = [I]*n
    for k in (range(len(i_list))) :
        if xyz[k] == 'z' : op_list[i_list[k]] = Z
        elif xyz[k] == 'x' : op_list[i_list[k]] = X
        elif xyz[k] == 'y' : op_list[i_list[k]] = Y
        else : 
            print("erreur lol choisis x y ou z")
            return None
        
    op1 = cste*op_list[0]
    for k in range(n-1) :
        op1 = scp.kron(op1,op_list[k+1], format='csr')
    return op1

def fond_ising(n, h) :
    """
    calcule l'état fondamental de l'hamiltonien d'une chaîne de n spins dans le modèle d'Ising

    Paramètres
    ----------
    n : int
        nombre de spins dans la chaîne 1D.
    h : float
        paramètre du champ magnétique
    
    Retourne
    -------
    float, ndarray (énergie propre et matrice de densité de l'état).

    """
    H = scp.csr_matrix((2**n,2**n)) # hamiltonien initialisé
    
    # Terme d'interaction x - x et Zeeman en z :
    for i in range(n) :
        if i != (n-1) : H += sigma(n,[i,i+1],'xx',-1)
        H += sigma(n,[i],'z',-h)
    H += sigma(n,[0,n-1],'xx',-1)
    vp = scp.linalg.eigsh(H, 1, which='SA')
    
    return vp[0], vp[1][:,0]

#################################################
#%% # fonctions extra :

def full_ising(n, h) :
    """
    calcule tous sauf l'état fondamental

    Paramètres
    ----------
    n : int
        nombre de spins dans la chaîne 1D.
    h : float
        paramètre du champ magnétique
    
    Retourne
    -------
    float, ndarray (énergie propre et matrice de densité de l'état).

    """
    H = scp.csr_matrix((2**n,2**n)) # hamiltonien initialisÃ©
    
    # Terme d'interaction x - x et Zeeman en z :
    for i in range(n) :
        if i != (n-1) : H += sigma(n,[i,i+1],'xx',-1)
        H += sigma(n,[i],'z',-h)
    H += sigma(n,[0,n-1],'xx',-1)
    vp = scp.linalg.eigsh(H, 2**n-1, which='LA')
    groundstates = [vp[1][:,i] for i in range(len(vp[0]))]
    
    return vp[0], groundstates

def fond_varXYZ(n, hx, hz, hxx, hyy, hzz) :
    """
    état fondamental de H = sum(hxx*XX + hyy*YY + hzz*ZZ + hx*X + hz*Z)
    
    Retourne
    -------
    float, ndarray (énergie propre et vecteur propre de l'état).
    
    """
    H = scp.csr_matrix((2**n,2**n)) # hamiltonien initialisé
    
    # Terme d'interaction xx, yy, zz et Zeeman en z :
    for i in range(n) :
        if i != (n-1) : 
            H += sigma(n,[i,i+1],'xx',-hxx)+sigma(n,[i,i+1],'yy',-hyy)+sigma(n,[i,i+1],'zz',-hzz)
        H += sigma(n,[i],'z',-hz) + sigma(n,[i],'x',-hx)
    H += sigma(n,[0,n-1],'xx',-hxx)+sigma(n,[0,n-1],'yy',-hyy)+sigma(n,[0,n-1],'zz',-hzz)
    vp = scp.linalg.eigsh(H, 1, which='SA')
    #print(H.toarray())
    return vp[0], vp[1][:,0]

def full_low_varXYZ(n, k, hx, hz, hxx, hyy, hzz) :
    """
    les k états de plus basse énergie de H = sum(hxx*XX + hyy*YY + hzz*ZZ + hx*X + hz*Z)
    
    Retourne
    -------
    liste de float, liste de ndarray (énergies propres et vecteurs propres des états).
    
    """
    H = scp.csr_matrix((2**n,2**n)) # hamiltonien initialisé
    
    # Terme d'interaction xx, yy, zz et Zeeman en z :
    for i in range(n) :
        if i != (n-1) : 
            H += sigma(n,[i,i+1],'xx',-hxx)+sigma(n,[i,i+1],'yy',-hyy)+sigma(n,[i,i+1],'zz',-hzz)
        H += sigma(n,[i],'z',-hz) + sigma(n,[i],'x',-hx)
    H += sigma(n,[0,n-1],'xx',-hxx)+sigma(n,[0,n-1],'yy',-hyy)+sigma(n,[0,n-1],'zz',-hzz)
    
    vp = scp.linalg.eigsh(H, k, which='SA')
    groundstates = [np.round(vp[1][:,i],15) for i in range(len(vp[0]))]
    
    return vp[0], groundstates

def full_low_XX(n, k, Jx, Jy, h) :
    """
    les k états de plus basse énergie du modèle XX
    
    Retourne
    -------
    liste de float, liste de ndarray (énergies propres et vecteurs propres des états).
    
    """
    return full_low_varXYZ(n, k, 0, h/2, Jx/4, Jy/4, 0)

#%% test 1212

jx= 0.5
py= 1
h = 1

val = full_low_XX(12, 1, jx, py*jx, h)
#val /= -val[0]
#plt.plot(range(20),val)
