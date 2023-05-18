from celmech.maps import CometMap
import numpy as np

muN = 5.15e-5
aN = 30.
PN = aN**(1.5)


def KAM_exists(cmap,Nensemble,Niter):
    ensemble = np.array([(0,0.5 + dw) for dw in np.random.normal(0,1e-5,Nensemble)])
    for i in xrange(Niter):
        for j in xrange(Nensemble):
            ensemble[j] = cmap(ensemble[j])
            if np.abs(ensemble[j,1]) > 1:
                return False
    else:
        return True

def experiment(q,Nensemble = 10,Ttraj = 5e7,return_cmap = False):
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



