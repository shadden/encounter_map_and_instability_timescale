from celmech.maps import CometMap
import rebound as rb
import numpy as np
from scipy.optimize import root_scalar


def get_M(T,sim,wrap=True):
    sim.integrate(T)
    orbits = sim.calculate_orbits(primary=sim.calculate_com())
    if wrap:
        return wrap2pi(orbits[1].M)
    return orbits[1].M
def get_sos_data(sim,Npts):
    a,phi = np.zeros((2,Npts))
    ps = sim.particles
    P = ps[2].P
    for i in range(Npts):
        orbits = sim.calculate_orbits(primary=sim.calculate_com())
        a[i] = orbits[1].a
        try:
            rt = root_scalar(get_M,args=(sim),bracket=[sim.t+0.45*P,sim.t+0.55*P])
        except:
            P = ps[2].P
            rt = root_scalar(get_M,args=(sim),bracket=[sim.t+0.45*P,sim.t+0.55*P])
        orbits = sim.calculate_orbits(primary=sim.calculate_com())
        phi[i] = wrap2pi(orbits[0].l-orbits[1].l)
        rt = root_scalar(lambda T: get_M(T,sim,wrap=False) - np.pi ,bracket=[sim.t+0.45*P,sim.t+0.55*P])
    return a,phi


def get_sim(m_pert,n_pert,a_tp,l_pert,l_tp,q_tp,pomega_tp):
    sim = rb.Simulation()
    sim.add(m=1)
    P_pert = 2 * np.pi / n_pert
    sim.add(m=m_pert,P=P_pert,l=l_pert)
    sim.add(m=0.,a = a_tp,l=l_tp,e=1-q_tp/a_tp,pomega=pomega_tp)
    sim.move_to_com()
    return sim

def get_jacobi_const(sim):
    ps = sim.particles
    star = ps[0]
    planet = ps[1]
    particle = ps[2]
    rstar = np.array(star.xyz)
    rplanet = np.array(planet.xyz)
    r = np.array(particle.xyz)
    v = np.array(particle.vxyz)
    
    KE = 0.5 * v@v # test particle kinetic energy
    mu1 = sim.G * star.m
    mu2 = sim.G * planet.m
    r1 = r-rstar
    r2 = r-rplanet
    PE = -1*mu1/np.sqrt(r1@r1) - mu2/np.sqrt(r2@r2) # test particle potential energy
    
    lz = np.cross(r,v)[-1]
    
    CJ = 2 * planet.n * lz - 2 * (KE + PE) # jacobi constant
    return CJ

def get_sim_at_fixed_CJ(a,CJ,q_bracket):
    get_sim_fn = lambda q,a: get_sim(m_pert,n_pert,a,l_pert,l_tp,q,pomega_tp)
    root_fn = lambda q,a: get_jacobi_const(get_sim_fn(q,a)) - CJ
    root = root_scalar(root_fn,args=(a,),bracket=q_bracket)
    assert root.converged, "Root-finding failed to converge for a={:.1f}, CJ={:.1f}".format(a,CJ)
    return get_sim_fn(root.root,a)

def get_map_pts(x0,cm,N):
    pts = np.zeros((N,2))
    for i in range(N):
        pts[i] = x0
        x0 = cm(x0)
    return pts

if __name__=="__main__":
    import matplotlib.pyplot as plt
    wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi)-np.pi
    wrap1 = lambda x: np.mod(x+0.5,2*np.pi)-0.5
    m_pert = 5.15e-5
    n_pert = 1
    l_pert = np.pi/2
    pomega_tp =  np.pi
    l_tp = pomega_tp + np.pi 
    Nres = 20
    q = 44.1 / 30.06

    cmap=CometMap(m_pert,Nres,q,kmax=32)
    fig,ax = plt.subplots(1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-0.5,0.5)
    for w0 in np.linspace(-0.5,0.5,endpoint=False):
        pts = get_map_pts((0.,w0),cmap,250)
        theta,w= pts.T
        theta = wrap2pi(theta)
        w = wrap1(w)
        ax.plot(theta/np.pi,w,'.',ms=0.5)

    plt.show()
