import numpy as np
from celmech.maps import CometMap
import sys


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

    savedir = "/fs/lustre/cita/hadden/03_comet_diffusion/escape_times3/"
    qs = np.linspace(35.5,38.5,5)
    Ns = range(11,17)

    i = int(sys.argv[1])
    j,k = i//6, i%6
    q = qs[j]
    N = Ns[k]
    cmap = CometMap(5.15e-5,N,q/30,max_kmax=64)

    Nensemble = 2**15
    Nstep = 100_000

    iters,w0s,bound  = get_local_escape_times(1e-5,Nensemble,100_000,cmap)

    Nlong = int(np.ceil( 1e-3 * Nensemble))
    w_long = w0s[np.argsort(iters)[-Nlong:]] 
    theta_long = np.zeros(Nlong)
    Nstep = int(np.max(iters[np.argsort(iters)[-Nlong:]]))

    orbits_long = np.zeros((Nstep,2,Nlong))
    for i in range(Nstep):
        orbits_long[i] = theta_long,w_long
        theta_long,w_long = cmap(np.array((theta_long,w_long)))
        

    w_random = np.random.choice(w0s,Nlong)
    theta_random = np.zeros(Nlong)
    orbits_random = np.zeros((Nstep,2,Nlong))
    for i in range(Nstep):
        orbits_random[i] = theta_random,w_random
        theta_random,w_random = cmap(np.array((theta_random,w_random)))
    
    np.savez(savedir + "long_vs_random_q{:.3f}_N{:03d}",orbits_long=orbits_long,orbits_random=orbits_random)

