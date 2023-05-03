import numpy as np
def get_map_pts_vs_time(pt0,mp,times,t=0):
    N = len(times)
    pts = np.zeros((N,2))
    for i,tf in enumerate(times):
        while t<tf:
            pt0 = mp(pt0)
            x = pt0[1]
            if x<0:
                return pts,t
            t+= x**(-1.5)
        pts[i] = pt0
    return pts,t

def get_init_points(x0,sigma_w0,Ntraj):
    w0s = np.random.normal(0,sigma_w0, Ntraj)
    N0 = x0**(-1.5)
    init_pts = np.array([
        (0,x0*(1 + 2*w/3/N0))
        for w in w0s
    ])
    return init_pts

def get_histories(init_pts,times,mp):
    histories = []
    for pt in init_pts:
        pts,_ = get_map_pts_vs_time(pt,mp,times)
        histories.append(pts)
    return np.array(histories)
from sos_utils import get_map_pts


def get_a_array(pt,mapfn,T,k=1):
    a = np.zeros(T)
    for i in range(T):
        x = pt[1]
        for _ in range(k):
            pt = mapfn(pt)
        a[i] = x - pt[1]
    return a

def compute_correlation_functions(mapfn,ensemble,T,k=1):
    """
    Compute the force correlation function, :math:`C_k`,
    for a map using an ensmble of initial points integrated
    for orbit segments of length :math:`L`.

    Arguments
    ---------
    mapfn : callable
        The map to iterate.
    ensemble : array-like
        The initial positions of ensemble members that will be used
        to esitate the correlation function.
    L : int
        The length of orbits to integrated
    """
    J = len(ensemble)
    D = np.zeros(J)
    C_arr = np.zeros((J,T))
    for i,pt in enumerate(ensemble):
        a = get_a_array(pt,mapfn,T,k)
        C_arr[i] = np.correlate(a,a,'full')[T-1:]
        C_arr[i] *= 1/(T - np.arange(T))
    return C_arr

def estimate_diffusion_coefficient(mapfn,ensemble,T,k=1):
    C_arr = compute_correlation_functions(mapfn,ensemble,T,k)
    Dmean = np.mean(np.cumsum(C_arr,axis=1),axis=0)
    Derr = np.sqrt(np.var(np.cumsum(C_arr,axis=1),axis=0)/len(ensemble))
    return Dmean, Derr

def estimate_diffusion_coefficient_simple(mapfn,ensemble,T):
    variances = np.zeros(T)
    for i in range(T):
        w = np.transpose(ensemble)[1]
        variances[i] = np.var(w)
        for j,pt in enumerate(ensemble):
            ensemble[j]=mapfn(pt)
    n = np.arange(T)
    D,_ = np.polyfit(n,variances,1)
    return D