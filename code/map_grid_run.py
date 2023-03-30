from three_planet_map import ThreePlanetMap
import numpy as np
import sys
from rebound.interruptible_pool import InterruptiblePool

# Map dynamical variables:
#   psi,M,psi21,pa,pe,L
#       pa = e^2 - da/a
#       pe = e^2
kmax = 5

def get_map(m1,m2,J1,J2,etp):
    n1 = J1/(J1-1)
    n2 = J2/(J2+1)
    alpha2 = n2**(2/3)
    eps1 = 2 * m1
    eps2 = 2 * m2 * alpha2
    J10 = np.round(J1)
    J20 = np.round(J2)
    plmap = ThreePlanetMap(eps1,eps2,J10,J20,etp,n1,n2,kmax)
    return plmap

def get_z0s(Ntp,etp):
    z0s = np.zeros((Ntp,6))
    z0s[:,:2] = np.random.uniform(-np.pi,np.pi,size=(Ntp,2))
    z0s[:,3:] = etp * etp
    return z0s

def record_trajectories(pl_map,Ntp,times):
    Ntimes = len(times)
    timesDone = np.zeros(Ntimes)
    a = np.zeros((Ntp,Ntimes))
    Tsyn = -2*np.pi/pl_map.n21
    dt = 0.2 * Tsyn
    etp = pl_map.etp
    z = get_z0s(Ntp,etp)
    ztrajs = np.zeros((Ntp,Nout,6))
    tnow = 0
    for i,t in enumerate(times):
        timesDone[i] = tnow
        for j in range(Ntp):
            ztrajs[j,i] = z[j]
            z[j],_tnow = pl_map.integrate(z[j],tnow,t,dt)
        tnow = _tnow
    return timesDone,ztrajs


def mapfunc(pars):
    m1,m2,J1,J2,etp,Ntp,Tfin,Nout,idnum = pars
    pl_map = get_map(m1,m2,J1,J2,etp)
    times = np.linspace(0,Tfin,Nout)
    timesDone,trajectories = record_trajectories(pl_map,Ntp,times)
    pa_vals = trajectories[:,:,3]
    pe_vals = trajectories[:,:,4]
    da_array = pe_vals - pa_vals
    np.save("da_{}".format(idnum),da_array)
    return da_array


Nout = 600
Tfin = 5e4 * 2 * np.pi
Ngrid = 50
Ntp = 10
m1 = m2 = 1e-5
etp = 0.09
J1s = 4 + np.linspace(-1,1,Ngrid)/2
J2s = 4 + np.linspace(-1,1,Ngrid)/2
pars = []
idnum = 0
for J2 in J2s:
    for J1 in J1s:
        pars.append((m1,m2,J1,J2,etp,Ntp,Tfin,Nout,idnum))
        idnum+=1

np.save("analytic_pars_array",np.array(pars))
pool = InterruptiblePool(16)
results = list(map(mapfunc,pars))

results_array = np.array(results).reshape(Ngrid,Ngrid,Ntp,Nout)
np.save("analytic_results_array",results_array)
