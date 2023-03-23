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
