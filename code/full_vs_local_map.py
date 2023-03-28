from celmech.maps import CometMap
import numpy as np
from sos_utils import get_map_pts
from matplotlib import pyplot as plt

wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi)-np.pi
cmap = CometMap(7e-5,20,1.3)

def fullmap(pt):
    theta,x = pt
    x1 = x - 2 * cmap.m * cmap.f(theta)
    theta1 = theta + 2 * np.pi / x1**(1.5)
    theta1 = np.mod(theta1,2*np.pi)
    return (theta1,x1)

Ntraj = 10
Npts = int(15e3)
init_pts = [(np.random.uniform(-.2,.2),w0)  for w0 in np.linspace(-0.5,0.5,Ntraj)]

fig,ax = plt.subplots(1,2,sharex=True)
x0 = cmap.N**(-2/3)
minw = np.inf
for pt in init_pts:
    # pts1 =get_map_pts(pt,cmap,Npts)
    # theta,w = np.transpose(pts1)
    # theta=wrap2pi(theta)
    # ax[0].plot(theta,w,'.',ms=0.5)

    pt_full = (pt[0],x0*(1+2 * pt[1]/cmap.N / 3))
    pts2 = get_map_pts(pt_full,fullmap,Npts)
    theta_full,x_full = np.transpose(pts2)
    theta_full=wrap2pi(theta_full)
    w_full = x_full**(-1.5) - cmap.N
    ax[0].plot(theta_full/np.pi,w_full,'.',ms=0.5)
    ax[1].plot(theta_full/np.pi,w_full,'.',ms=0.5)
    minw=min(np.min(w_full),minw)
ax[0].set_ylim(minw,minw+5)
ax[1].axhline(minw)
ax[1].axhline(minw+5)


plt.show()
