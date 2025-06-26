import numpy as np
from time import time
#from Aureliano_Buendia import *
#from Hspins import *

def Qtest(n, etat, kmin=2, kmax=4, if0=True) :
    """
    etat : le complet
    kmax : taille de sous-système maximale
    if0  : si on ne considère que les combianisons contenant 0
    """
    N_NPT = 0
    N_PPT = 0
    N_KFP = 0
    qubits = range(n)
    
    for k in range(kmin,kmax+1) :
        Qlist = combinations(qubits, k)
        for comb in Qlist :
            if ((if0==False) or 0 in comb) :
                rholil = trace_lil(n, np.array(etat), comb)
                for l in range(1, int(np.floor(k/2))+1) :
                    for jj in combinations(range(k), l) :
                        partition = [[comb[j] for j in jj], [comb[j] for j in range(k) if j not in jj]]
                        negativite = np.round(float(negat(k, rholil, jj)),10)
                        if negativite == 0:
                            KyFan = KFnorm(k, rholil, partition)
                            N_PPT +=1
                            if KyFan >= 1 : 
                                N_KFP += 1
                                print(f"n={n}, h={h}, partition={partition} !!!!!!!!!! {KyFan}")
                        else : N_NPT +=1
    #print(f'n={n}, h={h}','\nnombre de PPT :', N_PPT, '\nnombre de NPT :', N_NPT,
    #      "\nnombre d'états intriqués détectés par CCNR mais pas NPT :", N_KFP)

#%% ising full ???
start = time()

for n in range(2,6) :
    for h in np.logspace(-2,2,5, base=2) :
        etats = full_ising(n,h)[1]
        print('trouvé !')
        for etat in etats :
            Qtest(n,etat,kmax=n)

end = time()
print('\ntemps :', end-start, 'secondes')

#%% XX
start = time()

for n in range(11,13) :
    h,Jx,Jy = 1,1,1
    etat = full_low_XX(n, 1, Jx, Jy, h)[1]
    print('trouvé !')
    Qtest(n,etat,kmax=n)
    print(f"{n} qubits fini")

end = time()
print('\ntemps :', end-start, 'secondes')
