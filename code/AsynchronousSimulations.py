from celmech.maps import CometMap
from DiffusionExperimentUtils import *
import numpy as np
import sys
savedir = "./"
def get_trajectories_async(x0,sigma_w0,Ntraj,Tmax,comet_map, Nskip=1, Pplanet = 1 , safety_factor = 1):
    Nmax = int(np.ceil(safety_factor * Tmax * x0**(1.5) / Pplanet / Nskip))
    time = np.zeros(Ntraj)
    times = np.zeros((Nmax,Ntraj))
    xvals = np.zeros((Nmax,Ntraj))
    pts0 = np.transpose(get_init_points(x0,sigma_w0,Ntraj))

    i=0
    while np.min(time) < Tmax and i<Nmax:
        times[i] = time
        xvals[i] = pts0[1]
        i+=1
        for _ in range(Nskip):
            pts0 = comet_map.full_map(pts0)
            xs = pts0[1]
            time += Pplanet * xs**(-1.5)
    return times, xvals

TEST = True

muN = 5.15e-5
aN = 30.
qKAM1,NKAM1=np.transpose(np.load("../data/last_kam_curves2.npy"))
qKAM2,NKAM2=np.transpose(np.load("../data/last_kam_curves.npy"))
qs = np.concatenate((qKAM1,qKAM2))
Ncrits = np.concatenate((NKAM1,NKAM2))

I = int(sys.argv[1])
qI = int(sys.argv[2])

q = qs[qI]
N0 = Ncrits[qI] + 2
x0 = N0**(-2/3)

if TEST:
    Ntraj = 100
    Tfin  = 1e8
    Nskip = 100
else:
    Ntraj = 100
    Tfin  = 1e8
    Nskip = 100

cmap = CometMap(muN,40,q/aN)

PN = aN**(1.5)
sigma_w0 = 0.001

times, xvals = get_trajectories_async(x0,sigma_w0,Ntraj,Tfin,cmap,Nskip,Pplanet=PN)
np.savez_compressed('./{}/sim_q{:.2f}_async_nearXcrit_{:03d}'.format(savedir,q,I),times = times, x_values = xvals)
