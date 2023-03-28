from celmech.maps import CometMap
from DiffusionExperimentUtils import *
import numpy as np
import sys
TEST = False
I = int(sys.argv[1])

if TEST:
    Ntraj = 10
    Tfin  = 1e6
    Npts  = 10
else:
    Ntraj = 500
    Tfin  = 1e9
    Npts  = 10_000


muN = 5.15e-5
aN = 30.
q = 35.
a0 = 400.
PN = aN**(1.5)
sigma_w0 = 0.1

cmap = CometMap(muN,40,q/aN)
x0  = aN/a0
Pratio0 = x0**(-1.5)

times = np.linspace(0,Tfin / PN ,Npts)
init_pts =  get_init_points(x0,sigma_w0,Ntraj)
histories = get_histories(init_pts,times,cmap.full_map)
np.savez_compressed('./simresults/sim{:03d}'.format(I),times = PN * times, histories = histories)
