import numpy as np
from celmech.maps import CometMap
import sys
savedir = "/fs/lustre/cita/hadden/03_comet_diffusion/escape_times2/"

def get_local_escape_times(sigma_w0,Ntraj,itermax,comet_map, w_esc = 0.5):
    thetas = np.zeros(Ntraj)
    ws = 0.5 + sigma_w0 * np.random.randn(Ntraj)
    w0s = ws.copy()
    bound = np.ones(Ntraj,dtype = bool)
    iterations = np.zeros(Ntraj)
    Nbound = Ntraj
    i = 0
    while np.any(bound) and i<itermax:
        iterations[bound] = i
        thetas[bound],ws[bound] = comet_map(np.array((thetas[bound],ws[bound])))
        bound = np.logical_and(np.abs(ws-0.5)<w_esc,bound)
        i+=1
    return iterations, w0s, bound

if __name__=="__main__":
    datadir="/fs/lustre/cita/hadden/03_comet_diffusion/data/"
    qI = int(sys.argv[1])
    seed = int(sys.argv[2])
    qs = 37.5 + np.linspace(-1,1,5)
    np.random.seed(seed)
    q = qs[qI]
    N = 15
    cmap = CometMap(5.15e-5,N,q/30,max_kmax = 64)
    Ntraj = 2**15
    Nmax = 300_000
    iterations, w0s,  bound = get_local_escape_times(1e-3,Ntraj,Nmax,cmap)
    np.savez_compressed(
        '{}/local_escape_times_q{:.3f}_N{:03d}'.format(savedir,q,cmap.N),
        iterations = iterations,
        w0_values = w0s,
        bound = bound
    )


