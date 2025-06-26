import numpy as np
import scipy.sparse as scp
import mat73
from prettytable import PrettyTable


def psi(nH, état='0'):
    """
    psi: fonction qui d'éxciter un état donné sous forme vectoriel.
    :param nH: dicte l'espace d'Hilbert total qui sera de taille 2^nH,
            car nous travaillons avec des espaces pour des Qubits.

    :param état: dicte l'état excité en question par exemple 100 ou 011
            si nH =3.

    :return: la fonction retourne un array numpy de taille (2^nH,1)
    """

    vect = np.zeros((2 ** nH, 1))
    vect[int(état, 2)] = 1
    return vect


def dens(vect):
    '''
    dens: fonction qui crée une matrice de densité à partir d'un vecteur donné
    :param vect: vecteur d'état pur ou mixte
    :exception ValueError: ne fonctionne pas pour n>16
    :return: la matrice de densité d'état totale
    '''
    # norme = LA.norm(vect)
    état1 = scp.csr_matrix(vect)
    état2 = scp.csr_matrix(vect.T)
    rho = état1 @ état2
    # rho_norm = rho/norme**2
    return rho  # , rho_norm


def Tpartiel(rho, Qubits=[1]):
    '''
    Tpartiel: fonction qui effectue la transposée partielle sur le Qubit selectionné
    :param rho: matrice de densité
    :param Qubits: liste donnant les Qubits sur lesquelles on effectue la transposé
    :return: la matrice de densité transposé. elle décrit
    '''
    Qubits = np.array(Qubits) - 1
    dim = rho.shape[0]
    nQbit = int(np.log2(dim))

    rhoT = rho.copy()
    for i in range(dim):
        for j in range(dim):
            ai = format(i, f'0{nQbit}b')
            aj = format(j, f'0{nQbit}b')

            Lai = list(ai)
            Laj = list(aj)
            for q in Qubits:
                Laj[q], Lai[q] = Lai[q], Laj[q]

            i_prime = int(''.join(Lai), 2)
            j_prime = int(''.join(Laj), 2)
            rhoT[i_prime, j_prime] = rho[i, j]

    return rhoT


def nPartielT(rho, Qubits=[1]):
    '''
        nTpartiel: fonction qui effectue la transposée partielle sur le Qubit selectionné
        :param rho: matrice de densité
        :param Qubits: liste donnant les Qubits sur lesquelles on effectue la transposé
        :return: la matrice de densité transposé. elle décrit
        '''
    Qubits = np.array(Qubits)
    dim = rho.shape[0]
    nQbit = int(np.log2(dim))
    perm = np.arange(2 * nQbit)
    for s in Qubits:
        perm[s], perm[(s + int(nQbit))] = perm[(s + int(nQbit))], perm[s]
    return rho.reshape([2] * (nQbit*2)).transpose(perm).reshape(2 ** nQbit, 2 ** nQbit)


def PPT(rho):
    """
    Effectue le test PPT pour connaitre l'intrication d'un système
    :param rho: Une matrice transposé
    :returns: PPT/NPT Si le test est réussi, N la valeur de la négativite, les valeurs propres de la matrice

    """
    vp = np.linalg.eigvalsh(rho)
    valeur = vp
    # print(f'les valeur propre sont :\n{valeur}')
    N = (np.sum(np.abs(valeur) - valeur)) / 2

    # print(f'La négativité est de: {N:.4f}')
    if np.all(vp >= -1e-12):
        return 'PPT', N, valeur
    else:
        return 'NPT', N, valeur


def TracePart(rho, Qbit=[1]):
    Qbit = np.array(Qbit)
    dim = rho.shape[0]
    nQbit = int(np.log2(dim))

    bons = [n for n in range(nQbit) if n not in Qbit]
    dim_bons = 2 ** len(bons)
    rhoR = np.zeros((dim_bons, dim_bons))
    for i in range(dim):
        for j in range(dim):
            ai = format(i, f'0{nQbit}b')
            aj = format(j, f'0{nQbit}b')

            if all(ai[q] == aj[q] for q in Qbit):
                i_prime = int(''.join([ai[k] for k in bons]), 2)
                j_prime = int(''.join([aj[k] for k in bons]), 2)

                rhoR[i_prime, j_prime] += rho[i, j]
    return rhoR


def entropie(rho):  # applicable seulement si c'est l'état initial était pur
    """
    Calcule l'entropie de Von Neumann
    :param rho: la matrice réduite d'un système
    :return: la valeur d'entropie de Von Neumann
    """
    vp = np.linalg.eigvals(rho)
    vp = vp[vp.nonzero()[0]]
    entro = round(-np.sum(vp * np.log(vp)), 10)
    return entro


def mat_red(psi, GQbits):
    """
    Fonction faisant la matrice réduite selon
    :param psi: fonction psi pur à n Qubit
    :param GQbits: Qubits que l'on souhaite garder de la chaine de n Qubits
    :return: la matrice réduite du sous-système de nos Qubits
    """
    GQbits = np.array(GQbits)
    dim = psi.shape[0]
    nQbit = int(np.log2(dim))

    dim_tenseur = [2] * nQbit
    traces = [i for i in range(nQbit) if i not in GQbits]
    psi2 = psi.conj().T
    psi = psi.reshape(dim_tenseur)
    psi2 = psi2.reshape(dim_tenseur)

    rho_red = np.tensordot(psi, psi2, axes=(traces, traces))

    taille = 2 ** len(GQbits)

    rho_red = rho_red.reshape((taille, taille))
    return rho_red


def produit_tensoriel(liste):
    tenseur = liste[0]
    for i in liste[1:]:
        tenseur = scp.kron(tenseur, i, format='csr')
    return tenseur


def mat_til2(rho,m,n):
    if type(rho) == scp._csr.csr_matrix:
        rho = rho.toarray()
    rho = rho.reshape((m,n,m,n))
    R = rho.transpose((0,2,1,3))
    return R.reshape((m**2,n**2))

def ccnr2(mat,n, part):
    p0 = np.argmin([min(part[0]),min(part[1])])
    p1 = (p0+1)%2
    nA, nB = len(part[p0]), len(part[p1])

    if max(part[p0]) > min(part[p1]):
        for pp in part:
            pp = np.sort(pp)
        pflat = np.sort(list(part[0])+list(part[1]))
        indf = np.array([np.where(pflat == p)[0][0] for p in part[p0]])
        paires = []
        dist = 0

        for i in range((len(part[p0])-1)):
            dist += indf[i+1] - indf[i]-1
            if dist !=0 : paires = paires + [[i+1, i+1+dist]]
        if paires != [] :
            mat = perm(mat,n,paires)
    rho_tilde = mat_til2(mat,2**nA,2**nB)
    norme = np.linalg.svd(rho_tilde,compute_uv=False)
    return np.sum(norme)

def perm(mat,n,paires=[[0,1]]):
    permu = np.arange(2*n)
    for p_list in paires:
        permu[p_list[0]],permu[p_list[1]] = permu[p_list[1]],permu[p_list[0]]
        permu[p_list[0]+n],permu[p_list[1]+n] = permu[p_list[1]+n],permu[p_list[0]+n]
    return mat.reshape([2]*(2*n)).transpose(permu).reshape((2**n,2**n))


def permute_rho_by_qubit_order(rho, order: list):
    """
    Permute les qubits de la matrice rho selon un nouvel ordre donné.
    `order` est une permutation des indices [0, 1, ..., n-1]
    """
    n = int(np.log2(rho.shape[0]))
    assert rho.shape == (2**n, 2**n), "rho must be 2^n x 2^n"
    assert sorted(order) == list(range(n)), "order must be a permutation of qubit indices"

    # Construire la permutation sur les indices du tenseur (ket puis bra)
    perm = order + [i + n for i in order]

    rho_tensor = rho.reshape([2] * (2 * n))
    rho_perm = rho_tensor.transpose(perm).reshape((2**n, 2**n))
    return rho_perm

def indices_dans_liste(liste, sous_liste):
    return [i for i, val in enumerate(liste) if val in sous_liste]


# Benchmarking------------------------
def rho3x3(a):
    rho = 1 / (8 * a + 1) * np.array([[a, 0, 0, 0, a, 0, 0, 0, a],
                                      [0, a, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, a, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, a, 0, 0, 0, 0, 0],
                                      [a, 0, 0, 0, a, 0, 0, 0, a],
                                      [0, 0, 0, 0, 0, a, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1 / 2 * (1 + a), 0, 1 / 2 * np.sqrt(1 - a ** 2)],
                                      [0, 0, 0, 0, 0, 0, 0, a, 0],
                                      [a, 0, 0, 0, a, 0, 1 / 2 * np.sqrt(1 - a ** 2), 0, 1 / 2 * (1 + a)]])


    rhoT = 1 / (8 * a + 1) *np.array([[a, 0, 0, 0, 0, 0, 0,                   0, 0],
                                      [0, a, 0, a, 0, 0, 0,                   0, 0],
                                      [0, 0, a, 0, 0, 0, a,                   0, 0],
                                      [0, a, 0, a, 0, 0, 0,                   0, 0],
                                      [0, 0, 0, 0, a, 0, 0,                   0, 0],
                                      [0, 0, 0, 0, 0, a, 0,                   a, 0],
                                      [0, 0, a, 0, 0, 0, 1/2*(1+a),           0, 1/2*np.sqrt(1-a**2)],
                                      [0, 0, 0, 0, 0, a, 0,                   a, 0],
                                      [0, 0, 0, 0, 0, 0, 1/2*np.sqrt(1-a**2), 0, 1/2*(1+a)]])

    return rho,rhoT
def bell(*args):
    psi0 = np.array([[1,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [1,0,0,1]])*0.5*args[0]
    psi1 = np.array([[0,0,0,0],
                     [0,1,1,0],
                     [0,1,1,0],
                     [0,0,0,0]])*(0.5)*args[1]

    psi2 = np.array([[0,0,0,0],
                     [0,1,-1,0],
                     [0,-1,1,0],
                     [0,0,0,0]])*0.5*args[2]
    psi3 = np.array([[1,0,0,-1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [-1,0,0,1]])*.5*args[3]

    return psi0 + psi1 + psi2 + psi3

def rhoF(nh,F):
    psiplus = np.zeros(2**(2*nh))
    for i in range(2**nh):
        psiplus[i*(2**nh+1)]=1
    d = nh**2
    psiplus = psiplus.reshape(psiplus.shape[0],1)
    rhoplus = (1/d)*dens(psiplus)
    I = np.eye(2**(2*nh))
    rf = (1-F)/(d**2-1)*(I-rhoplus)+F*rhoplus

    return rf
'''
#-----------------------------------------------------------------------
#%% si l'on veut un état mixte peut prendre un système de nQbit et faire la trace pour
# avoir une mixture de projecteur et on a un état mixte!
W = psi(3,'001') + psi(3,'010') + psi(3,'100')
GHZ = (psi(2,'01') - psi(2,'10'))/np.sqrt(2)

rho,_ = dens(GHZ)
rho_T = Tpartiel(rho,[1])
rho_A = TracePart(rho,cell=1)
entro = entropie(rho_A)
print(f'La matrice de densité de l\'état psi est:\n {rho} \n')
print(f'Sa matrice réduite selon A est:\n{rho_A}\n')
print(f'Son entropie de Von Neumann de {entro}k_b\n')
print(f'Sa transposé partielle selon A est de :\n{rho_T}\n')
print(PPT(rho_T))

#%% Le modèle d'ising
IT0h1 = mat73.loadmat('ising_T=0.0_h=1.0.mat')["rho"]
IT0h05 = mat73.loadmat('ising_T=0.0_h=0.5.mat')['rho']
IT05h1 = mat73.loadmat('ising_T=0.5_h=1.0.mat')["rho"]
IT1h1 = mat73.loadmat('ising_T=1.0_h=1.0.mat')['rho']
IT0h1_A_BC = Tpartiel(IT0h1, [1])
IT0h1_B_AC = Tpartiel(IT0h1, [2])
IT0h05_A_BC = Tpartiel(IT0h05,[1])
IT0h05_B_AC = Tpartiel(IT0h05,[2])
IT05h1_A_BC = Tpartiel(IT05h1,[1])
IT05h1_B_AC = Tpartiel(IT05h1,[2])
IT1h1_A_BC = Tpartiel(IT1h1,[1])
IT1h1_B_AC = Tpartiel(IT1h1,[2])

table = PrettyTable(['T','h','N_A_BC','N_B_AC'])
table.add_row(['0.0','1.0',f'{PPT(IT0h1_A_BC)[1]:.4f}',f'{PPT(IT0h1_B_AC)[1]:.4f}'])
table.add_row(['0.0','0.5',f'{PPT(IT0h05_A_BC)[1]:.4f}',f'{PPT(IT0h05_B_AC)[1]:.4f}'])
table.add_row(['0.5','1.0',f'{PPT(IT05h1_A_BC)[1]:.4f}',f'{PPT(IT05h1_B_AC)[1]:.4f}'])
table.add_row(['1.0','1.0',f'{PPT(IT1h1_A_BC)[1]:.4f}',f'{PPT(IT1h1_B_AC)[1]:.4f}'])

print(table)

#%% Chaine Kagome

Kagome = mat73.loadmat('kagome_CSL_triangle.mat')['rho']
Kagome_A_BC = Tpartiel(Kagome,[1])

print(PPT(Kagome_A_BC))


#travail sur hamiltonirn de ising quantique

'''


#IMPORTANT POUR LES PERMUTATIONS ON DEVRAIT FAIRE (1,0,2,3,4,5,6,7,8) PAR EXEMPLE. PAS OBLIGE DE MATRICE
