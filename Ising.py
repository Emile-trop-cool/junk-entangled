import matplotlib
import numpy as np
from travail import *

matplotlib.use('TkAgg')

from fonctions import *
import scipy.sparse as scp
from scipy.sparse.linalg import eigsh
import time
from itertools import combinations
import matplotlib.pyplot as plt
import toqito
from toqito.state_props import is_ppt


start = time.time()

X = scp.csr_matrix([[0,1],
                    [1,0]])
Y = np.array([[0,-1j],
              [1j,0]])
Z = scp.csr_matrix([[1,0],
                    [0,-1]])
I = scp.eye(2, format='csr')


def ising_0(n,h):
    """
    Fonction qui calcule l'état fondamental de l'Hamiltonien d'une chaine de n Qubits
    en 1D selon le modèle d'Ising.
    :param n: nombre de Qubits de la chaine
    :return: matrice de densité de l'état
    """
    H=scp.csr_matrix((2**n,2**n))

    for j in range(n):
        liste_x = [I]*n
        liste_x[j] = X
        liste_x[(j+1)%n] = X
        H-= produit_tensoriel(liste_x)

        liste_z = [I]*n
        liste_z[j] = Z
        H-= h*produit_tensoriel(liste_z)
    return H

def ising_1(n,h,J):
    H = scp.csr_matrix((2**n,2**n))

    for j in range(n):
        liste_y = [I]*n
        liste_y[j] = Y
        liste_y[(j+1)%n] = Y
        H-= J*produit_tensoriel(liste_y)/4

        liste_x = [I] * n
        liste_x[j] = X
        liste_x[(j + 1) % n] = X
        H -= J*produit_tensoriel(liste_x)/4

        liste_z = [I] * n
        liste_z[j] = Z
        H -= h * produit_tensoriel(liste_z)/2

    return H


n=22
nb =4
Qubit2 = ising_0(n,0.5)
val,vecteur = eigsh(Qubit2,k=1,which='SA')
vecteur = np.round(vecteur,12)

test012 = mat_red(vecteur,[1,2,3,5])
#mat_trans = Tpartiel(test012,[1,2])

def test_ccnr(rho4,soussys):
    leA = soussys
    leB = [i for i in range(nb) if i not in leA]
    norme = ccnr2(rho4,nb,[leB,leA])
    return norme

def testa():
    Comb = combinations([i for i in range(n)], nb)

    nPPT = 0
    nNPT = 0
    CcNr = 0


    for comb in Comb:
        if 0 in comb:

            test = mat_red(vecteur,comb)

            for j in range(nb): #pour 1vs 3
                T13 = nPartielT(test,[j])
                ppt = PPT(T13)
                LN = np.log2(ppt[1] * 2 + 1)
                #print(LN)
                #print(comb[j], comb,[i for i in comb if i not in [comb[j]]],min(min((np.array([i for i in comb if i not in [comb[j]]])-comb[j])%n)-1,min((np.array([i for i in comb if i not in [comb[j]]])*(n-1)+comb[j])%n)))
                if ppt[0] == 'NPT':
                    nNPT+=1
                elif ppt[0] == 'PPT':
                    nPPT+=1
                    #print(comb[j], comb)
                    KF = test_ccnr(test,[j])
                    if KF>= 1:
                        CcNr +=1
                    print(comb[j],comb)
                    indicess = indices_dans_liste(comb,[i for i in comb if i not in [comb[j]]])
                    ordre =  [j]+ indicess
                    #print(ordre)

                    Ntest = permute_rho_by_qubit_order(test,ordre)
                    #print(Ntest)
                    pol1 = RandomBlochPolytope(2,200)
                    #print(pol1)
                    le1v3 = RobustnessToSeparabilityByBlochPolytope(Ntest,pol1,paires=[[j],indicess],num_iter=30)
                    print(le1v3[0],'\n')


                    #print(comb[j],comb,'\n')
            #for co in combinations(comb,2): # pour 2vs2
            for jj in combinations(range(nb),int(nb/2)):
                soussys = list(jj)
                T22 = nPartielT(test,soussys)
                ppt2 = PPT(T22)
                LN = np.log2(ppt2[1]*2+1)
                #print(LN)
                if ppt2[0] == 'NPT':
                    nNPT+=1
                if ppt2[0] == 'PPT':
                    nPPT+=1
                    KF2 = test_ccnr(test,soussys)
                    print(soussys,comb)
                    combi = [comb[i] for i in jj]
                    indicess2 = indices_dans_liste(comb,[i for i in comb if i not in combi])
                    ordre2 =  jj+ indicess2
                    Ntest2 = permute_rho_by_qubit_order(test,ordre2)

                    polypoly = RandomBlochPolytope(4,300)
                    le2vs2 = RobustnessToSeparabilityByBlochPolytope(Ntest2,polypoly,paires=[jj,indicess2],num_iter=30
                                                                     ,convergence_accuracy= 1e-04)
                    print(le2vs2[1],'\n')

                    if KF2 >=1 :
                        CcNr +=1
                #print(comb)
                #print(ppt[0], round(ppt[1], 4))
                #print([comb[j] for j in jj],comb,'\n')

            '''for jjj in combinations(range(nb), int(nb / 3)): #dans le cas de 6 qubits, 2vs4
                soussyss = list(jjj)
                T24 = nPartielT(test, soussyss)
                ppt24 = PPT(T24)
                LN = np.log2(ppt2[1] * 2 + 1)
                # print(LN)
                if ppt24[0] == 'NPT':
                    nNPT += 1
                if ppt24[0] == 'PPT':
                    nPPT += 1
                    KF24 = test_ccnr(test, soussyss)
                    print(soussyss, comb)
                    # polypoly = RandomBlochPolytope(8,300)
                    # le2vs2 = RobustnessToSeparabilityByBlochPolytope(T22,polypoly,[[0,1,2],[3,4,5]],30,convergence_accuracy= 1e-04)
                    # print(le2vs2[1],'\n')

                    if KF24 >= 1:
                        CcNr += 1'''
    return nNPT,nPPT,CcNr

nNPT, nPPT,CcNr = testa()
print(nNPT,nPPT,CcNr)
end = time.time()
print("--- %f seconds ---" % (end-start))



testois = eigsh(ising_0(4,0.5),k=1,which='SA')
#print(testois,'\n',np.round(dens(testois[1]).toarray(),10))
#print(np.round(testas,10))
#rhooo,rhoooT = rho3x3(0.236)
#print(PPT(rhoooT))
#print(np.sum(ccnr(mat_tilde(rhooo,3,3))))

#État diagonal de Bell
#Belle = bell(0.1,0.7,.1,.1)
#print(np.sum(ccnr(mat_tilde(Belle,2,2))))

#état isotropique
#rof = rhoF(2,1/5)
#print(np.sum(ccnr(mat_tilde(rof,4,4))))



#H0 = ising_0(4,.5)
#H1 = ising_1(4,.4, .5)
#hval, hvec = eigsh(H1,k=1,which='SA')


def isstoq(mat):
    if not scp.issparse(mat):
        raise TypeError('La matrice doit être sparse')
    if (mat-mat.T).nnz != 0 :
        return False
    elif not np.isreal(mat.data).all():
        return False

    diag_i = np.arange(mat.shape[0])
    off_diag = mat.copy()
    off_diag[diag_i,diag_i]=0

    return np.all(off_diag.data <=0 )


#test precis pour certain couple de valeur 3v3

comb = (0,2,4,8,10,12)
rdm = mat_red(vecteur,comb)
soussys = [0,1,2]

#arad = RobustnessToSeparabilityByBlochPolytope(rdm,RandomBlochPolytope(8,300),paires=[[0,1,2],[3,4,5]],num_iter=30)

