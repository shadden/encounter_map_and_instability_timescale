import numpy as np
import rebound as rb
import sys
from rebound.interruptible_pool import InterruptiblePool

def get_sim(m1,m2,J1,J2,etp,Ntp):
    n1 = J1/(J1-1)
    n2 = J2/(J2+1)
    sim=rb.Simulation()
    sim.add(m=1)
    sim.add(m=m1,P=2*np.pi/n1,l=0)
    sim.add(m=m2,P=2*np.pi/n2,l=0)
    for i in range(Ntp):
        name="pid{}".format(i)
        sim.add(m=0,a=1,e=etp,l='uniform',pomega='uniform',hash=name)
    sim.move_to_com()
    sim.integrator='whfast'
    sim.ri_whfast.safemode=0
    sim.dt = sim.particles[1].P / 20.
    return sim

def record_a_and_e(sim,times):
    Ntp = sim.N-3
    Ntimes = len(times)
    a = np.zeros((Ntp,Ntimes))
    e = np.zeros((Ntp,Ntimes))
    ps = sim.particles
    for i,t in enumerate(times):
        sim.integrate(t)
        for j in range(Ntp):
            p=ps['pid{}'.format(j)]
            a[j,i]=p.a
            e[j,i]=p.e
    return a,e

def mapfunc(pars):
    m1,m2,J1,J2,etp,Ntp,Tfin,Nout,idnum = pars
    sim = get_sim(m1,m2,J1,J2,etp,Ntp)
    sim.automateSimulationArchive("./sim_id{:d}.sa".format(idnum),interval=Tfin/Nout)
    sim.integrate(Tfin)

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
np.save("pars_array",np.array(pars))
pool = InterruptiblePool(16)
results = pool.map(mapfunc,pars)
