import numpy as np
from celmech.maps import CometMap
from sos_utils import *
import rebound as rb

m_pert = 5.15e-5
n_pert = 1
Nres = 20
q = 44.1 / 30.06

l_pert = 0.0
pomega_tp =  np.pi
l_tp = pomega_tp + np.pi 

from celmech.maps import CometMap
cmap = CometMap(m_pert,Nres,q,kmax=32)
fig,ax=plt.subplots(1,2,sharey=True,sharex=True,figsize=(15,5))

Ngrid = 20
w0s = np.linspace(-0.5,0.5,Ngrid)
theta0s = np.linspace(-np.pi,np.pi,Ngrid,endpoint=False)
wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi) - np.pi

sim0=get_sim(m_pert,n_pert,Nres**(2/3),l_pert,l_tp,q,pomega_tp)
CJ0 = get_jacobi_const(sim0) 

sims = []
for i,w0 in enumerate(w0s):
    Ptp = cmap.N + w0
    atp = Ptp**(2/3)
    for j,theta0 in enumerate(theta0s):
        l_pert = wrap2pi(theta0 - n_pert * Ptp * 0.5)
        sim = get_sim_at_fixed_CJ(atp,CJ0,[0.9*q,1.2*q],m_pert,n_pert,l_pert,np.pi,0)
        sims.append(sim)

try:
    data = np.load("./all_sos_pts.npy")
except:    
    all_sos_pts = []
    for i,sim in enumerate(sims):
        all_sos_pts.append(get_sos_data(sim,200))
    data = np.array(all_sos_pts)
    np.save("./all_sos_pts",data)

for y,x in data:
    l,=ax[0].plot(x,y**(1.5),'.',ms=0.75)
    pts = get_map_pts((x[0],y[0]**(1.5) - cmap.N),cmap,200)
    phi = wrap2pi(pts[:,0])
    x = pts[:,1]
    ax[1].plot(phi,cmap.N+x,'.',ms=0.75,color=l.get_color())

for a in ax:
    a.set_xlim(-np.pi,np.pi)
    a.set_ylim(19.5,20.5)
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
#plt.show()
plt.savefig("../figures/comet_nbody_vs_map.png")




    
