import matplotlib
import numpy as np

matplotlib.use('TkAgg')

from fonctions import *
import scipy.sparse as scp
from scipy.sparse.linalg import eigsh
import time
from itertools import combinations
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
    lst=[]

    for j in range(n):
        liste_x = [I]*n
        liste_x[j] = X
        liste_x[(j+1)%n] = X
        H-= produit_tensoriel(liste_x)

        liste_z = [I]*n
        liste_z[j] = Z
        H-= h*produit_tensoriel(liste_z)
    return H
n=14
Qubit2 = ising_0(n,.5)
val,vecteur = eigsh(Qubit2,k=1,which='SA')
vecteur = np.round(vecteur,12)

test012 = mat_red(vecteur,[1,3,4,5])
#mat_trans = Tpartiel(test012,[1,2])


Comb = combinations([i for i in range(n)],4)

nPPT = 0
nNPT = 0
CcNr =0

def test_ccnr(rho4,soussys):
    leA = soussys
    leB = [i for i in range(4) if i not in leA]
    norme = ccnr2(rho4,4,[leB,leA])
    return norme

def test():
    Comb = combinations([i for i in range(n)], 4)

    nPPT = 0
    nNPT = 0
    CcNr = 0


    for comb in Comb:
        test = mat_red(vecteur,comb)

        for j in range(4): #pour 1vs 3
            T13 = nPartielT(test,[j])
            ppt = PPT(T13)
            if ppt[0] == 'NPT':
                nNPT+=1
            elif ppt[0] == 'PPT':
                nPPT+=1
                KF = test_ccnr(test,[j])
                if KF>= 1:
                    CcNr +=1
                    print(comb[j],comb)


                #print(comb[j],comb,'\n')
        #for co in combinations(comb,2): # pour 2vs2
        for jj in combinations(range(4),2):
            soussys = list(jj)
            T22 = nPartielT(test,soussys)
            ppt2 = PPT(T22)
            if ppt2[0] == 'NPT':
                nNPT+=1
            if ppt2[0] == 'PPT':
                nPPT+=1
                KF2 = test_ccnr(test,soussys)
                print(soussys,comb)
                if KF2 >=1 :
                    CcNr +=1
                    print(soussys, comb,KF2,'\n')
            #print(comb)
            #print(ppt[0], round(ppt[1], 4))
            #print([comb[j] for j in jj],comb,'\n')
print(nNPT,nPPT,CcNr)
   # negat,_,_ = PPT(Trans)
    #print(negat,comb)
#mat = dens(vecteur)
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

#test22 = dens(eigsh(ising_0(10,0.5),k=1,which='SA')[1])
#print(mat_tilde(te#st22,32,32))
#aaaaaaa = np.load('evil_matilde.npy')
#testeEmile = mat_tilde(dens(eigsh(ising_0(10,.5),k=1,which='SA')[1]),5,5)

#print(np.sum(mat_tilde(test22,32,32)-aaaaaaa))


#%% #convention says its brabra-ketket


