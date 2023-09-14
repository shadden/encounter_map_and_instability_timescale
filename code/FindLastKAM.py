from celmech.maps import CometMap
import numpy as np

muN = 5.15e-5
aN = 30.
PN = aN**(1.5)

def KAM_exists(cmap,Nensemble,Niter):
    ensemble = np.transpose([(0,0.5 + dw) for dw in np.random.normal(0,1e-5,Nensemble)])
    for i in xrange(Niter):
        ensemble = cmap(ensemble)
        w = ensemble[1]
        if not np.alltrue(np.logical_and(w<1,0<w)):
            return False
    else:
        return True

def experiment(q,Nensemble = 1_000,Ttraj = 5e9,return_cmap = False):
    cmap = CometMap(muN,10,q/aN,max_kmax = 64,rtol = 0.07)
    epscrit = cmap.get_eps_crit(kmax = 2)
    Ncrit = int(np.floor((epscrit/cmap.m/3)**(3/5)))
    cmap.N = Ncrit
    Ntraj = int( Ttraj / PN / cmap.N)
    while not KAM_exists(cmap,Nensemble,Ntraj):
        cmap.N -= 1
        Ntraj = int( Ttraj / PN / cmap.N)
    if return_cmap:
        return q,cmap.N,cmap
    
    return q,cmap.N



