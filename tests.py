from travail import *
from Ising import *

hamil = ising_0(8,.5)

def save_object(obj,filename=''):
    try:
        with open(f"data{filename}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def lowerbound(n, rho, precision=4) :
    tlist = np.linspace(0,1,11)
    for j in range(1,precision+1) :
        for t in tlist :
            rhot = t*rho + ((1-t)/2**n)*np.eye(2**n)
            neg = PPT(nPartielT(rhot,[0,1]))[1]
            if neg != 0 :break
            tmax = t
        if tmax == 1 : break
        tlist = np.linspace(tmax,tmax+10**(-j),10,False)
    return tmax+10**(-1*(precision))



m=12

for _ in range(100):
    etatpur = np.random.randn(2 ** 12) + 1j * np.random.randn(2 ** 12)
    etatpur /= np.linalg.norm(etatpur)
    rhored = mat_red(etatpur, [0, 1, 2, 3, 4, 5])

    lb = np.round(lowerbound(6, rhored, 6), 6)
    pt = np.round(RobustnessToSeparabilityByBlochPolytope(rhored, RandomBlochPolytope(8, 140), paires=[[0,1,2],[3,4,5]],num_iter=40, convergence_accuracy=1e-06)[0], 6)
    print(f"t max pour être PPT : {lb}, t max du polytope : {pt}"
          + f"\ndifférence (PPT-poly) : {np.round(lb - pt, 7)}")
    if np.round(lb-pt,5) !=0:
        print(rhored)
        save_object(rhored, f'rhoe3{_}')