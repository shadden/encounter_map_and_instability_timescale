import numpy as np
from scipy.optimize import root_scalar
import rebound as rb

def get_map_pts(x0,cm,N):
    pts = np.zeros((N,2))
    for i in range(N):
        pts[i] = x0
        x0 = cm(x0)
    return pts

wrap2pi = lambda x: np.mod(x+np.pi,2*np.pi)-np.pi
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
        
        # record a at apocenter...
        a[i] = orbits[1].a
        
        # ... then record theta at pericenter
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

def get_sim_at_fixed_CJ(a,CJ,q_bracket,m_pert,n_pert,l_pert,l_tp,pomega_tp):
    get_sim_fn = lambda q,a: get_sim(m_pert,n_pert,a,l_pert,l_tp,q,pomega_tp)
    root_fn = lambda q,a: get_jacobi_const(get_sim_fn(q,a)) - CJ
    root = root_scalar(root_fn,args=(a,),bracket=q_bracket)
    assert root.converged, "Root-finding failed to converge for a={:.1f}, CJ={:.1f}".format(a,CJ)
    return get_sim_fn(root.root,a)

def wisdomF_and_G(omega,Omega,t):
    t = np.mod(t,2*np.pi/Omega)
    wt = omega*t
    pi_om_Om = np.pi * omega / Omega
    arg1 = wt   - pi_om_Om
    denom = Omega * np.sin(pi_om_Om) / np.pi
    d_denom = np.cos(pi_om_Om)
    cos_arg = np.cos(arg1)
    sin_arg = np.sin(arg1)
    F =  cos_arg/denom - 1 / omega
    G = -sin_arg/denom
    if t==0:
        G = 0 
    dF = 1/omega/omega - cos_arg * d_denom / denom / denom - (t - np.pi/Omega)*sin_arg / denom
    dG = sin_arg * d_denom / denom / denom - (t - np.pi/Omega)*cos_arg / denom
    return F,G,dF,dG


if __name__=="__main__":
    from celmech.maps import StandardMap, CometMap
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    YLIM=0.1
    cmap = CometMap(3e-5,20,1.4,kmax=16)
    
    for t0 in np.linspace(-np.pi,np.pi,endpoint=False):
        pts = get_map_pts((t0,0),cmap,200)
        theta,p = pts.T
        theta = np.mod(theta+np.pi,2*np.pi)-np.pi
        ax[0].plot(theta,p,'k.',ms=0.5)
        ax[1].plot(theta,p,'k.',ms=0.5)

    theta0 = np.linspace(-np.pi,np.pi,149,endpoint=False)    
    w0 = np.linspace(-YLIM,YLIM)
    theta0,w0 = np.meshgrid(theta0,w0)

    H0 = 0.5*(w0)**2
    for kk,amp in enumerate(cmap.amps):
        k=kk+1
        Ck = amp/k
        H0+= 0.5* cmap.eps * Ck * np.cos(k*theta0) / np.pi

    ax[0].contour(theta0,w0,H0,levels=20)
    ax[0].set_ylim(-YLIM,YLIM)

    w1 = w0.copy()
    theta1 = theta0.copy()
    for kk,amp in enumerate(cmap.amps):
        k = kk+1
        Ck = amp/(k+1)
        Ak = 0.5 * cmap.eps * Ck  / np.pi
        F,G,dF,dG = wisdomF_and_G(k*w0,1,0.999*2*np.pi)
        theta1 += Ak*(dF*np.sin(k*theta0) + dG * np.cos(k*theta0)) 
        w1 += Ak*(F * np.cos(k*theta0) - G * np.sin(k*theta0))
    H1 = 0.5*(w1)**2
    for kk,amp in enumerate(cmap.amps):
        k=kk+1
        Ck = amp/k
        H1+= 0.5 * cmap.eps * Ck * np.cos(k*theta1) / np.pi
    ax[1].contour(theta0,w0,H1,levels=20)
    plt.show()
if False:
    import matplotlib.pyplot as plt
    K = 0.75
    fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

    smap = StandardMap(K)
    for p0 in np.linspace(-2,2):
        pts = get_map_pts((np.pi,p0),smap,200)
        theta,p = pts.T
        theta = np.mod(theta+np.pi,2*np.pi)-np.pi
        ax[0].plot(theta,p,'k.',ms=0.5)
        ax[1].plot(theta,p,'k.',ms=0.5)

    theta0 = np.linspace(-np.pi,np.pi,149,endpoint=False)    
    P0 = np.linspace(-3,3)
    theta0,P0 = np.meshgrid(theta0,P0)
    H0 = 0.5*P0**2 + K*np.cos(theta0)
    ax[0].contour(theta0,P0,H0,levels=20)
    ax[0].set_ylim(-2,2)

    F,G,dF,dG = wisdomF_and_G(P0,2*np.pi,0.9999)
    P1 = P0 + K * (F * np.cos(theta0) - G * np.sin(theta0))
    theta1 = theta0 - K * (dF*np.sin(theta0) + dG * np.cos(theta0))
    H1 = 0.5*P1**2 + K*np.cos(theta1)
    ax[1].contour(theta0,P0,H1,levels=20)

    
    theta0 = np.linspace(-np.pi,np.pi,149,endpoint=False)    
    for s in [-1,1]:
        P0 = s*np.sqrt(2*K*(1-np.cos(theta0)))
        F,G,dF,dG = wisdomF_and_G(P0,2*np.pi,0.9999)
        P1 = P0 - K * (F * np.cos(theta0) - G * np.sin(theta0))
        theta1 = theta0 + K * (dF*np.sin(theta0) + dG * np.cos(theta0))
        ax[0].plot(theta0,P0,color='k')
        ax[1].plot(theta1,P1,color='r')
    plt.show()

