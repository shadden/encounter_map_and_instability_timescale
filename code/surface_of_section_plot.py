import numpy as np
from celmech.maps import CometMap
from sos_utils import get_map_pts, get_sim, get_jacobi_const

mN = 5.15e-5
nN = 1
lN = np.pi/2
pmg_tp = np.pi
# https://iopscience.iop.org/article/10.3847/1538-4357/ac866b/pdf
Nres = 20
q = 44.1/30.06

sim0=get_sim(m_pert,n_pert,Nres**(2/3),l_pert,l_tp,q,pomega_tp)
CJ0 = get_jacobi_const(sim0)

a0s = (Nres*(1+np.linspace(0,.02,2*16)))**(2/3) 
sims = []
for a0 in a0s:
    sims.append(get_sim_at_fixed_CJ(a0,CJ0,[q*0.9,q*1.2]))
try:
    data = np.load("./all_sos_pts.npy")
except:    
    all_sos_pts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i,sim in enumerate(sims):
            all_sos_pts.append(get_sos_data(sim,200))
    data = np.array(all_sos_pts)
    np.save("./all_sos_pts",data)

from matplotlib import pyplot as plt

fig,ax=plt.subplots(1,2,sharey=True,sharex=True,figsize=(15,5))

wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi) - np.pi
for y,x in data:
    l,=ax[0].plot(x,y**(1.5),'.',ms=1)
    pts = get_map_pts((x[0],y[0]**(1.5) - cmap.N),cmap,500)
    phi = wrap2pi(pts[:,0])
    x = pts[:,1]
    ax[1].plot(phi,cmap.N+x,'.',ms=0.5,color=l.get_color())

for a in ax:
    a.set_xlim(-np.pi,np.pi)
    a.set_xticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi])
    a.set_xticklabels(["$-\pi$","$-\pi/2$","$0$","$\pi/2$","$\pi$"],fontsize=15)
    a.set_xlabel(r"$\theta$",fontsize=16)
    xax=a.xaxis
    xax.set_tick_params(direction='in',width=2,size=8)
    yax= a.yaxis
    yax.set_tick_params(direction='in',width=2,size=8,labelsize=15)
plt.subplots_adjust(wspace=0.05)
ax[0].set_title("$N$-body; $q=44.1$ AU",fontsize=16)
ax[1].set_title("Map; $q=44.1$ AU",fontsize=16)
ax[0].set_title("$N$-body; $q=44.1$ AU",fontsize=16)
ax[0].set_ylabel("$P/P_N$",fontsize=16)
plt.tight_layout()
#plt.savefig("../figures/comet_nbody_vs_map.png")
plt.show()