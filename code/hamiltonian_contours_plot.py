from matplotlib import pyplot as plt
import numpy as np
from celmech.maps import CometMap
from scipy.optimize import root_scalar

wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi)-np.pi

def hamiltonian_function(theta,w,j,k,cmap):
    w0 =  j/k - cmap.N 
    p = w-w0
    if k==1:
        return 0.5*p*p + 0.5 * cmap.eps * np.vectorize(cmap.F)(k*theta) / np.pi
    else:
        H = 0.5*p*p 
        lmbda = cmap.lambda_const
        PE = -0.5 * cmap.A_const * np.log(2 * (np.cosh(k * lmbda) - np.cos(k * theta))) / k
        for i in range(k,cmap.kmax+1,k):
            delta_ck = cmap.delta_ck[i]
            PE += delta_ck * np.cos(i * theta)
        H+= 0.5 * cmap.eps * PE / np.pi
        return H
        
Fsequence = [(1,4),(1,3),(2,5),(1,2),(3,5),(2,3),(3,4)]
if __name__=="__main__":

    mN = 5.15e-5
    Nmmr = 10
    q = 44.1
    aN = 30
    atp = aN * Nmmr**(2/3)
    mN = 5.15e-5
    cmap = CometMap(mN,Nmmr,q/aN,max_kmax=64)

    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))
    Norbit = 200
    Ngrid = 400

    for n,d in Fsequence:
        theta0s = np.linspace(np.pi/d,0,6)
        Hfn = lambda x,y: hamiltonian_function(x,y,d * cmap.N + n ,d,cmap)
        w0=n/d
        Esx = Hfn(0,w0)
        rt = root_scalar(lambda z: Hfn(np.pi/d,z)-Esx,x0=w0,x1 = w0+0.1)
        wmax = rt.root
        dw = wmax-w0
        th,w = np.linspace(-np.pi,np.pi,Ngrid),w0 + dw * np.linspace(-1,1,Ngrid)
        X,Y = np.meshgrid(th,w)
        Z = Hfn(X,Y)
        Ecirc = Hfn(np.pi/d,w0)
        Esx = Hfn(0,w0)
        lvls=np.sort([Hfn(theta0,w0) for theta0 in theta0s])
        ax[0,1].contour(X/np.pi,Y,Z,levels=lvls,colors=['k'],linestyles=['-'])
        
        for theta0 in theta0s:
            pt = (theta0,w0)
            orbit = np.zeros((Norbit,2))
            for i in range(Norbit):
                orbit[i]=pt
                pt = cmap(pt)
            theta,w = np.transpose(orbit)
            theta=wrap2pi(theta)
            ax[0,0].plot(theta/np.pi,w,'.',ms=1)

            
    for n,d in [(0,1),(1,1)]:
        theta0s = np.linspace(np.pi/d,0,6)
        Hfn = lambda x,y: hamiltonian_function(x,y,d * cmap.N + n ,d,cmap)
        w0=n/d
        Esx = Hfn(0,w0)
        rt = root_scalar(lambda z: Hfn(np.pi,z)-Esx,x0=w0,x1 = w0+0.1)
        wmax = rt.root
        dw = wmax-w0
        th,w = np.linspace(-np.pi,np.pi,Ngrid),w0 + dw * np.linspace(-1.5,1.5,Ngrid)
        X,Y = np.meshgrid(th,w)
        Z = Hfn(X,Y)
        lvls=np.sort([Hfn(theta0,w0) for theta0 in theta0s])
        ax[0,1].contour(X/np.pi,Y,Z,levels=lvls,colors=['k'],linestyles=['-'])
        for theta0 in np.linspace(-np.pi,np.pi,13,endpoint=False):
            pt = (theta0,w0)
            orbit = np.zeros((Norbit,2))
            for i in range(Norbit):
                orbit[i]=pt
                pt = cmap(pt)
            theta,w = np.transpose(orbit)
            theta=wrap2pi(theta)
            ax[0,0].plot(theta/np.pi,w,'.',ms=1)

    ax[0,0].set_ylim(0,1)
    ax[0,0].set_ylabel("$w$",fontsize=16)
    
    ax[0,0].set_xlabel(r"$\theta/\pi$",fontsize=16)
    ax[0,1].set_xlabel(r"$\theta/\pi$",fontsize=16)

    ax[0,0].set_title(r"Map",fontsize=18)
    ax[0,1].set_title(r"Analytic",fontsize=18)
    for a in ax:
        plt.sca(a)
        plt.tick_params(direction='in',size=8,labelsize=14)
        
    plt.tight_layout()
    plt.savefig("../figures/map_vs_analytic.png")