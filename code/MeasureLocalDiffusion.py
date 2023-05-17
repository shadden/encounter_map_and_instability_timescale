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

def ak(theta,w,k,cmap):
    for _ in range(k+1):
        theta_old,w_old = theta,w
        theta,w = cmap((theta,w))
    return w-w_old

if __name__=="__main__":
    from scipy.fft import rfft2
    from scipy.interpolate import interp1d
    import sys
    
    # Parameters
    mN = 5.15e-5 # Mass of Neptune
    Ngrid = 128 # grid size for fft
    Niter = 1000 # number of iterations for numerical measurement of diffusion rate
    Nensemble = 1000 # size of ensemble for numerical measurement of diffusion rate
    kmax = 4 # maximum k value to use for computing analytic diffusion rate

    # Set percienter distance from command line argument
    I = int(sys.argv[1])
    q = np.linspace(35,65,7)[I]/30.0

    # Last KAM curve data
    kam_data = np.load("../data/last_kam_curves.npy")
    q_kam,Ncrit = np.transpose(kam_data)
    Nmin = int(interp1d(q_kam/30,Ncrit)(q)) + 30 # min MMR
    Nmax = Nmin +  200 # max MMR
    Nstep = 2 # step size in MMRs

    Nvals = np.arange(Nmin,Nmax,Nstep) # MMRs to loop through

    # initialize map
    cmap = CometMap(mN,Nmin,q)
    
    THETA,W = np.meshgrid(
        np.linspace(0,2*np.pi,Ngrid,endpoint=False),
        np.linspace(0,1,Ngrid,endpoint=False)
    )
    
    eps0 = cmap.eps
    result_0 = rfft2(np.vectorize(ak)(THETA,W,0,cmap),norm="forward")

    D_num = np.zeros(len(Nvals))
    k_vals = np.arange(2,kmax+1,dtype=int)
    Ck_vals = np.zeros((kmax+1,len(Nvals)))

    for i,Nval in enumerate(Nvals):
        ensemble = get_ensemble((0,0.5),1e-3,1000)
        cmap.N = Nval
        var = VarianceVersusIteration(ensemble,Niter,cmap)
        D,_=np.polyfit(np.arange(Niter),var,1)
        D_num[i] = D
        Ck_vals[0,i] = cmap.D_QL()
        for k in k_vals:
            result_k = rfft2(np.vectorize(ak)(THETA,W,k,cmap),norm="forward")
            Ck_vals[k,i] = 2 * np.real((cmap.eps/eps0) * 2 * result_0[0] @ np.conjugate(result_k[0]))
    
    np.save("../data/numerical_diffusion_values_{}".format(I),D_num)
    np.save("../data/analytic_Ck_values_{}".format(I),Ck_vals)
    np.save("../data/N_values_{}".format(I),Nvals)
    