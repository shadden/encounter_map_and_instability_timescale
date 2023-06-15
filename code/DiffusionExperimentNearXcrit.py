from celmech.maps import CometMap
from DiffusionExperimentUtils import *
import numpy as np
import sys
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
    Ntraj = 10
    Tfin  = 1e6
    Npts  = 10
else:
    Ntraj = 500
    Tfin  = 5e9
    Npts  = 5_000

cmap = CometMap(muN,40,q/aN)

PN = aN**(1.5)
sigma_w0 = 0.001

times = np.linspace(0,Tfin / PN ,Npts)
init_pts =  get_init_points(x0,sigma_w0,Ntraj)
histories = get_histories(init_pts,times,cmap.full_map)
np.savez_compressed('./simresults/sim_q{:.2f}_nearXcrit_{:03d}'.format(q,I),times = PN * times, histories = histories)
