import rebound as rb
import numpy as np

NORB = 50
def get_simulation(pars,aNep = 30,mNep=5.15e-5):
    a,q = pars
    etp = 1-q/a
    sim = rb.Simulation()
    sim.units = ("Msun","AU","yr")
    sim.integrator = 'whfast'
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1)
    sim.add(m=mNep,a=aNep)
    sim.add(m=0,a=a,e=etp,M = np.pi)
    sim.dt = sim.particles[1].P / 25.
    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 2 * a
    return sim

def MEGNO_run(sim,Norb):
    tp = sim.particles[2]
    Ptp = tp.P
    times = sim.t + np.linspace(0,Norb * Ptp)
    escape = 0 
    try:
        for i,t in enumerate(times):
            sim.integrate(t)
            MEGNO = sim.calculate_megno()
            if MEGNO > 20:
                break
    except rb.Escape:
        MEGNO = sim.calculate_megno()
        escape = 1
    ly = sim.calculate_lyapunov()
    return MEGNO,ly, escape


def MEGNO_from_aq(pars):
    sim = get_simulation(pars)
    Y,ly,escape = MEGNO_run(sim,NORB)
    return Y,ly,escape
    