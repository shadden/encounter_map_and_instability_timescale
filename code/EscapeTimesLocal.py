import numpy as np
from celmech.maps import CometMap
import sys

savedir = "./"

def get_local_escape_times(sigma_w0,Ntraj,itermax,comet_map, w_esc = 1):
    thetas = np.zeros(Ntraj)
    ws = 0.5 + sigma_w0 * np.random.randn(Ntraj)
    bound = np.ones(Ntraj,dtype = bool)
    iterations = np.zeros(Ntraj)
    Nbound = Ntraj
    i = 0
    while np.any(bound) and i<itermax:
        iterations[bound] = i
        thetas[bound],ws[bound] = comet_map(np.array((thetas[bound],ws[bound]))) 
        bound = np.logical_and(np.abs(ws)<w_esc,bound)
        i+=1
    return iterations, ws, thetas, bound

if __name__="__main__":
    qKAM1,NKAM1=np.transpose(np.load("../data/last_kam_curves2.npy"))
    qKAM2,NKAM2=np.transpose(np.load("../data/last_kam_curves.npy"))
    qs = np.concatenate((qKAM1,qKAM2))

    I = int(sys.argv[1])
    qI = int(sys.argv[2])
    dN = int(sys.argv[3])

    Ncrits = np.concatenate((NKAM1,NKAM2))
    q = qs[qI]
    N0 = Ncrits[qI] 

    cmap = CometMap(5.15e-5,N0 + dN,q/35,max_kmax = 64)

    iterations, ws, thetas, bound = get_local_escape_times(1e-3,4096,1_000_000,cmap)
    np.savez_compressed(
        '{}/local_escape_times_q{:.2f}_dN{:03d}_{:03d}'.format(savedir,q,I),
        iterations = iterations, 
        w_values = ws,
        theta_values = thetas,
        bound = bound
    )


