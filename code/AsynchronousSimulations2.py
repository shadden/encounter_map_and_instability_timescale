from celmech.maps import CometMap
from DiffusionExperimentUtils import *
import numpy as np
import sys
savedir = "./"
def get_trajectory_escape_times(x0,sigma_w0,Ntraj,Tmax,comet_map, Pplanet = 1 , x_esc = 0):
    times = np.zeros(Ntraj)
    thetas,xs = np.transpose(get_init_points(x0,sigma_w0,Ntraj))
    bound = np.ones(Ntraj,dtype = bool)
    while np.any(bound) and np.any(times[bound]<Tmax):
        times[bound] += Pplanet * xs[bound]**(-1.5)
        thetas[bound],xs[bound] = comet_map.full_map(np.array((thetas[bound],xs[bound])))
        bound = np.logical_and(xs>x_esc,bound)
    return times, xs, thetas, bound

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
    Ntraj = 16
    Tmax  = 1e8
else:
    Ntraj = 512
    Tmax  = 1e10

cmap = CometMap(muN,40,q/aN,max_kmax = 64)

PN = aN**(1.5)
sigma_w0 = 0.001
times, xs, thetas, bound = get_trajectory_escape_times(x0,sigma_w0,Ntraj,Tmax,cmap, Pplanet = PN , x_esc = (N0+1)**(-2/3))

print("{:d} of {:d} particles survived until {:.0f}".format(np.sum(bound),Ntraj,Tmax))
for time in np.sort(times):
    print("{:.2f}".format(time/1e6))
np.savez_compressed(
    './{}/sim_q{:.2f}_exit_times_{:03d}'.format(savedir,q,I),
    times = times, 
    x_values = xs,
    theta_vales = thetas,
    bound = bound
    )
