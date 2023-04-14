from celmech.maps import CometMap
import numpy as np
from sos_utils import get_map_pts
from sympy import totient

def get_ensemble(pt0,sigma_w,Npts):
    theta0,w0 = pt0
    w0s = np.random.normal(w0,sigma_w,Npts)
    pts = np.array([(theta0,w) for w in w0s])
    return pts
def VarianceVersusIteration(ensemble,Niter,mapfn):
    var = np.zeros(Niter)
    for i in range(Niter):
        var[i] = np.var(ensemble[:,1])
        ensemble = np.transpose(mapfn(ensemble.T))
    return var

def second_order_overlap_eps(cmap):
    tot = 0
    first_order_half_width_sq = 0
    for k_minus_one,amp in enumerate(cmap.amps):
        k=k_minus_one+1
        ck = amp/k
        if k==2:
            half_width = np.sqrt(2 * ck / np.pi)
            tot+=2*totient(k)*half_width
        if k%2:
            # odd orders contribute to first-order 
            # width at pi
            first_order_half_width_sq += 4 * (0.5/np.pi) * ck
    tot += np.sqrt(first_order_half_width_sq)
    return 1/tot/tot

mN = 5.15e-5
N = 20
beta_vals = np.linspace(30/65.,30/35.,10)
cmap = CometMap(mN,20,40./30.)
crits1,crits2 = np.zeros((2,len(beta_vals)))
for i,beta in enumerate(beta_vals):
    if beta<1/1.8:
        cmap._kmax = 12
    else:
        cmap._kmax = 32
    cmap.q = 1/beta

    crits1[i] = cmap.get_eps_crit()
    crits2[i] = second_order_overlap_eps(cmap)

from matplotlib import pyplot as plt
fig = plt.figure()
for ecrit in [crits1,crits2]:
    a = (ecrit/(3*cmap.m))**(2/5)
    q = 30/beta_vals
    plt.plot(a,q,'s-')
plt.xscale('log')
plt.show()



if False:
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(2)
    Nf = 50
    Dratios = np.zeros(Nf)
    fvals = np.geomspace(8,1000,Nf)
    for i,f in enumerate(fvals):
        cmap.m  = f * (epscrit / cmap.eps) * cmap.m
        ensemble  = get_ensemble((0,0.5),1e-8,100)
        var = VarianceVersusIteration(ensemble,500,cmap)
        imin = np.sum(var<1e-3)
        N = np.arange(len(var))
        x,y=N[imin:],var[imin:]
        imin = np.sum(var<1e-2)
        x,y=N[imin:],var[imin:]
        ax[0].plot(x,y)
        slope,intercept = np.polyfit(x,y,1)
        Dratios[i] = slope/cmap.D_QL()
    ax[1].plot(fvals,Dratios,'ks-')
    ax[1].set_xscale('log')
    plt.show()


