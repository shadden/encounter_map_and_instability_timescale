import matplotlib.pyplot as plt
from sos_utils import get_map_pts
from celmech.maps import CometMap
import numpy as np

wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi)-np.pi
cmap = CometMap(7e-5,20,1.3)

def fullmap(pt):
    theta,x = pt
    x1 = x - 2 * cmap.m * cmap.f(theta)
    theta1 = theta + 2 * np.pi / x1**(1.5)
    theta1 = np.mod(theta1,2*np.pi)
    return (theta1,x1)

def advance_point_to_time(pt,t0,tfin):
    t = t0
    while t<tfin:
        pt = fullmap(pt)
        P = (pt[1])**(-1.5)
        t += P
    return pt,t

if __name__=="__main__":
    Ntraj = 500
    Ntimes = 10
    times = np.linspace(0,1e5,Ntimes)

    x0 = cmap.N**(-2/3)
    pts = [
        (np.random.uniform(-.2,.2),x0 * (1+2 * w0/cmap.N / 3))  
        for w0 in np.linspace(-0.1,0.1,Ntraj)
    ]
    times_done = np.zeros(Ntraj)
    pts = np.array(pts)
    all_pts = np.zeros((Ntimes,Ntraj,2))
    for i,time in enumerate(times):
        for j in range(Ntraj):
            pts[j],times_done[j] = advance_point_to_time(pts[j],times_done[j],time)
        all_pts[i] = pts
    
    fig,ax = plt.subplots(1)
    xmin,xmax= np.min(all_pts[:,:,1]),np.max(all_pts[:,:,1])
    bins= np.linspace(xmin,xmax,20)
    for pts in all_pts[1:]:
        ax.hist(pts[:,1],bins=bins,histtype='step',lw=3,density=True)
    plt.show()