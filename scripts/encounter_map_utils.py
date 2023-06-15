import numpy as np
from celmech.maps import EncounterMap
from scipy.optimize import minimize_scalar
import rebound as rb
from warnings import warn

### Get stable/unstable manifolds ###
from celmech.maps import solve_manifold_f_and_g,manifold_approx,func_from_series
def get_manifold_approx_error(Tprime,u,Nmax,farr,garr):
    """
    """
    p0,p1_approx = manifold_approx(u,Nmax,farr,garr)
    p1 = Tprime(p0)
    error = np.linalg.norm((p1-p1_approx))/np.linalg.norm(p1)
    return error

def solve_stable_unstable_manifold(emap,xunst,Nsegment,Nmax=11,Npts=1000):
    """
    Generate stable and unstable manifold segments.

    Arguments
    ---------
    emap : celmech.EncounterMap
        Map to compute manifolds for
    xunst : ndarray, shape (2,)
        Unstable fixed point (should be (pi,0))
    Nsegment : int
        How many times to iterate initial segments
    Nmax : int 
        Order of Taylor-series approximation for initial segment
    Npts : int
        Number of points to use to resolve initial segment.
    """
    emap.mod = False
    answer = []
    for i,unstable in enumerate([True,False]):
        if unstable:
            mapfn = emap
        else:
            mapfn = emap.inv
        R,farr,garr = solve_manifold_f_and_g(xunst,emap,Nmax,unstable=unstable)
        Tprime = lambda p: R@(mapfn(xunst + R.T@p) - xunst)
        minresult = minimize_scalar(lambda u: get_manifold_approx_error(Tprime,u,Nmax,farr,garr),bracket=[-7,-1])
        u0=minresult.x
        u1 = func_from_series(garr,u0)
        uvals = np.linspace(u0,u1,Npts)
        pts,_ = manifold_approx(uvals,Nmax,farr,garr)
        theta,p = pts
        for n in range(Nsegment):
            pts = np.transpose([Tprime(p) for p in pts.T])
            theta = np.concatenate((theta,pts[0]))
            p = np.concatenate((p,pts[1]))
        theta,p = R.T @ [theta,p]
        theta += xunst[0]
        p += xunst[1]
        answer.append((theta,p))
    return answer
def plot_with_breaks(xvals,yvals,xmod,ymod,ax,center_x = False,center_y=True,**kwargs):
    """
    Plot a curve on a torus assuming xvals are taken modulo xmod
    and yvals are taken modulo ymod.
    """
    if center_x:
        wrapx = lambda z: np.mod(z+0.5*xmod,xmod)-0.5*xmod
    else:
        wrapx = lambda z: np.mod(z,xmod)
    if center_y:
        wrapy = lambda z: np.mod(z+0.5*ymod,ymod)-0.5*ymod
    else:
        wrapy = lambda z: np.mod(z,ymod)
    xvals = wrapx(xvals)
    yvals = wrapy(yvals)
    breaksX = np.abs(xvals[1:] - xvals[:-1])>0.5*xmod
    breaksY = np.abs(yvals[1:] - yvals[:-1])>0.5*ymod
    break_indices = np.arange(len(xvals)-1)[np.logical_or(breaksX,breaksY)]
    break_indices=np.concatenate(([0],break_indices,[-1]))
    for ilow,ihi in zip(break_indices[:-1],break_indices[1:]):
        result = ax.plot(xvals[ilow+1:ihi],yvals[ilow+1:ihi],**kwargs)
    return result

### Get N-body surface of section ###
def get_sim(m_pert,n_pert,a_tp,l_pert,l_tp,e_tp,pomega_tp):
    """
    Get a rebound simulation of the planar restricted three-body problem.
    """
    sim = rb.Simulation()
    sim.add(m=1)
    P_pert = 2 * np.pi / n_pert
    sim.add(m=m_pert,P=P_pert,l=l_pert)
    sim.add(m=0.,a = a_tp,l=l_tp,e=e_tp,pomega=pomega_tp)
    sim.move_to_com()
    return sim
def get_psi(T,sim):
    r"""
    Compute :math:`\psi = \lambda_1 - \lambda_2` at time T
    from a rebound simulation
    """
    ps = sim.particles
    sim.integrate(T)
    return np.mod(ps[1].l - ps[2].l ,2*np.pi)
def get_jacobi_const(sim):
    r"""
    Calculate the Jacobi constant from a rebound
    simulation of the restricted 3BP. Assumes
    particle 1 is the planet, particle 2 is the
    test particle.
    """
    ps = sim.particles
    star = ps[0]
    planet = ps[1]
    particle = ps[2]
    rstar = np.array(star.xyz)
    rplanet = np.array(planet.xyz)
    r = np.array(particle.xyz)
    v = np.array(particle.vxyz)

    ke = 0.5 * v@v
    mu1 = sim.G * star.m
    mu2 = sim.G * planet.m
    r1 = r-rstar
    r2 = r-rplanet
    pe = -1*mu1/np.sqrt(r1@r1) - mu2/np.sqrt(r2@r2)

    lz = np.cross(r,v)[-1]

    CJ = 2 * planet.n * lz - 2 * (ke + pe)
    return CJ

from scipy.optimize import root_scalar
def get_sim_at_fixed_CJ(CJ,m_pert,n_pert,a,l_pert,l_tp,e_bracket,pomega_tp):
    r"""
    Generate a rebound simulation with Jacobi constant C_J by solving for
    the particle eccentricity in ``e_bracket=[e_min,e_max]``.
    """
    get_sim_fn = lambda e,a: get_sim(m_pert,n_pert,a,l_pert,l_tp,e,pomega_tp)
    root_fn = lambda e,a: get_jacobi_const(get_sim_fn(e,a)) - CJ
    root = root_scalar(root_fn,args=(a,),bracket=e_bracket)
    assert root.converged, "Root-finding failed to converge for a={:.1f}, CJ={:.1f}".format(a,CJ)
    return get_sim_fn(root.root,a)
def get_sos_data_staggered(sim,Npts,nmin=0,nmax=np.inf):
    r"""
    Get surface of section points, recording test-particle
    mean motion and eccentricity at \psi = pi and test particle
    mean anomaly at \psi = 0.

    Arguments
    ---------
    sim : rebound simulation
        Simulation to compute SOS points for.
    Npts : int
        Number of points to compute.
    nmin : float, optional
        If test particle mean motion falls below nmin,
        integration is terminated and points are returned.
    nmax : float, optional
        If test particle mean motion goes above nmax,
        integration is terminated and points are returned.

    Returns
    -------
    n : array
        Array of test particle mean motion at opposition.
    e : array
        Array of test particle eccentricity at opposition.
    M : array
        Array of test particle mean anomaly at conjunction.
    """
    ps = sim.particles
    n_syn = np.abs(ps[1].n - ps[2].n)
    Tsyn = 2 * np.pi / n_syn
    n,e,M = np.zeros((3,Npts))
    wrap2pi= lambda x: np.mod(x+np.pi,2*np.pi)-np.pi
    for i in range(Npts):
        try:
            rt = root_scalar(lambda t: wrap2pi(get_psi(t,sim)) , bracket=[sim.t + 0.45*Tsyn,sim.t + 0.55*Tsyn])
            M[i] = ps[2].M
            rt = root_scalar(lambda t: wrap2pi(get_psi(t,sim) - np.pi) , bracket=[sim.t + 0.45*Tsyn,sim.t + 0.55*Tsyn])
            n[i] = ps[2].n
            e[i] = ps[2].e
        except:
            # re-compute Tsyn
            n_syn = ps[1].n - ps[2].n
            Tsyn = 2*np.pi/n_syn
            rt = root_scalar(lambda t: wrap2pi(get_psi(t,sim)) , bracket=[sim.t + 0.3*Tsyn,sim.t + 0.7*Tsyn])
            M[i] = ps[2].M
            rt = root_scalar(lambda t: wrap2pi(get_psi(t,sim) - np.pi) , bracket=[sim.t + 0.3*Tsyn,sim.t + 0.7*Tsyn])
            n[i] = ps[2].n
            e[i] = ps[2].e
        if n[i]<nmin or n[i]>nmax:
            warn("Integration terminated early because paricle mean motion\
                 exceeded limits")
            return n[:i],e[:i],M[:i]
    return n,e,M
