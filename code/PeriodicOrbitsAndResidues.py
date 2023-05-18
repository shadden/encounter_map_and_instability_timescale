import numpy as np
from scipy.optimize import root_scalar, root
EPS = np.finfo(float).eps
rtEPS = np.sqrt(EPS)
from warnings import warn

class NewtsafeResult():
    def __init__(self,root,newton_calls=0,biscetions=0,converged=False):
        self.root = root
        self.newton_calls=newton_calls
        self.bisections=biscetions
        self.conveged = converged

def newtsafe(fDf,a,b,max_iter=20,xtol=rtEPS,**kwargs):
    r"""
    Implementation of a `safe' Newton's method root search.
    Tries to uses Newton's method at each step but reverts to 
    bisection when the proposed root is outside the bracketing
    interval.

    Parameters
    ----------
    fDf : function
        A function returning both the function value, :math:`f(x)` and
        its first derivative, :math:`f'(x)`.
    a : float
        One edge of the initial bracketing interval.
    b : float
        Other edge of the initial bracketing interval.
    max_iter : int, optional
        Maximum number of iterations, by default 20
    xtol : float, optional
        Tolerance for root accuracy, by default the square root of
        machine precision

    Returns
    -------
    NewtsafeResult
        Representation of the root finding result

    Raises
    ------
    ValueError
        Raises value error if function does not change sign on 
        initial interval points.
    """
    fl,Dfl = fDf(a)
    fh,Dfh = fDf(b)
    if fl*fh > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if np.isclose(fl,0,atol=xtol):
        return a
    if np.isclose(fh,0,atol=xtol):
        return b
    # orient search so f(a)<0<f(b)
    xl,xh = (a,b) if  fl<0 else (b,a)
    root = NewtsafeResult(0.5 * (xl + xh))
    dx_old = np.abs(xh-xl)
    dx = dx_old
    f,Df = fDf(root.root)
    for i in range(max_iter):
        # proposed root in interval?
        cond1 = ((root.root-xh)*Df - f)*((root.root-xl)*Df - f) > 0
        # decreasing fast enough?
        cond2 = np.abs(2*f) > np.abs(dx_old*Df)
        if cond1 or cond2:
            # bisect
            root.bisections+=1
            dxold = dx
            dx = 0.5 * (xh - xl)
            root.root = xl + dx
            if np.isclose(root.root,xl,atol=xtol,rtol=0):
                root.conveged = True
                return root
        else:
            root.newton_calls+=1
            dxold = dx
            dx = f/Df
            temp = root.root
            root.root = root.root - dx
            if np.isclose(root.root,temp,atol=xtol,rtol=0):
                root.conveged = True
                return root
        f,Df=fDf(root.root)
        if f<0:
            xl = root.root
        else:
            xh = root.root
    else:
        warn("Exceeded maximum number of iterations ({})".format(max_iter))
        return root


def nest_map_list(pt0,cmap,n):
    pts = np.zeros((n,2))
    for i in range(n):
        pts[i] = pt0
        pt0 = cmap(pt0)
    return pts

def nest_map(pt0,cmap,n):
    for _ in range(n):
        pt0 = cmap(pt0)
    return pt0
def nest_map_with_variational(pt0,Dpt0,cmap,n):
    for _ in range(n):
        pt0,Dpt0 = cmap.with_variational(pt0,Dpt0)
    return pt0,Dpt0

def find_periodic_orbit(m,n,mapfn,minimax=False,**kwargs):
    
    mapfn.mod = False
    symmetry_line1,symmetry_line2,symmetry_line3 = mapfn.symmetry_lines()
    if minimax:
        symline = symmetry_line1
    else:
        symline = symmetry_line2 if n%2 else symmetry_line3
    rootfn = lambda s: nest_map(symline(s),mapfn,n)[0] - symline(s)[0] - 2 * np.pi * m
    x0 = kwargs.setdefault('x0',2 * m * np.pi / n)
    xx = 2 * (m-1) * np.pi / n
    x1 = x0+ 0.5*(xx - x0)
    kwargs.setdefault('x1',x1)
    rt = root_scalar(rootfn,**kwargs)
    pt0 = symline(rt.root)    
    orbit = nest_map_list(pt0,mapfn,n)
    
    return orbit

def _root_function_2d(pt0,m,n,mapfn):
    jac = np.eye(2)
    pt = pt0.copy()
    for _ in range(n):
        pt = mapfn(pt)
        jac = mapfn.jac(pt) @ jac
    return pt - pt0 - np.array((2*np.pi*m,0)), jac - np.eye(2)

def _find_a_periodic_orbit_2d(m,n,guess,mapfn,**kwargs):
    rt = root(_root_function_2d,guess,args=(m,n,mapfn),jac=True,**kwargs)
    assert rt.success, "Root finding failed for orbit ({},{}) starting at point {}".format(m,n,guess)
    pt0 = rt.x
    orbit = nest_map_list(pt0,mapfn,n)
    return orbit

def _find_a_periodic_orbit(m,n,mapfn,symline,**kwargs):
    if kwargs.get('method')=='newton':
        rootfn = lambda s: _newton_root_function(s,m,n,mapfn,symline)
        kwargs['fprime'] = True
        rt = root_scalar(rootfn,**kwargs)
    elif kwargs.get('method')=='newtsafe':
        rootfn = lambda s: _newton_root_function(s,m,n,mapfn,symline)
        a,b = kwargs['bracket']
        rt = newtsafe(rootfn,a,b,**kwargs) 
    else:
        rootfn = lambda s: nest_map(symline(s),mapfn,n)[0] - symline(s)[0] - 2 * np.pi * m
        rt = root_scalar(rootfn,**kwargs)
    pt0 = symline(rt.root)    
    orbit = nest_map_list(pt0,mapfn,n)

    # Handle potential period-doubling bifurcations that 
    # produce new pairs of points on symmetry lines
    # s0 = rt.root
    # sf = mapfn(orbit[-1])[1]
    # if not np.isclose(s0,sf,atol=1e-7):
    #     s1,s2 = np.sort((s0,sf))
    #     ds = s2 - s1
    #     kwargs['bracket'] = (s1 +0.01*ds,s2-0.01*ds)
    #     kwargs['method'] = 'brentq'
    #     orbit =  _find_a_periodic_orbit(m,n,mapfn,symline,**kwargs)
    return orbit

def _newton_root_function(s,m,n,mapfn,symline):
    pt0 = symline(s)
    tangent_dir = symline(1.) - symline(0)
    tangent_dir /= np.linalg.norm(tangent_dir)
    pt,Dpt = nest_map_with_variational(pt0,tangent_dir,mapfn,n)
    x = pt[0] - pt0[0] - 2 * np.pi * m
    dx = Dpt[0] - tangent_dir[0]
    return x, dx


def find_periodic_orbits_all(m,n,mapfn,brackets=None,x0s=None,x1s=None,**kwargs):
    mapfn.mod = False
    symmetry_line1,symmetry_line2,symmetry_line3 = mapfn.symmetry_lines()
    symline = symmetry_line1
    
    # Defaults for secant method search
    if x0s is None:
        x0 = 2 * m * np.pi / n
        x0s = [x0 for _ in range(3)]
    if x1s is None:
        x0 = 2 * m * np.pi / n
        xx = 2 * (m-1) * np.pi / n
        x1s = [x0+ 0.5*(xx - x0) for _ in range(3)]
    if brackets is None:
        brackets = [None for _ in range(3)]
    i = 0 
    residues = np.zeros(3)
    actions = np.zeros(3)
    orbits = np.zeros((3,2,n))
    symline_points = np.zeros(3)
    for symline,x0,x1,bracket in zip(mapfn.symmetry_lines(),x0s,x1s,brackets):
        kwargs['x0'] = x0
        kwargs['x1'] = x1
        kwargs['bracket'] = bracket
        orbit = _find_a_periodic_orbit(m,n,mapfn,symline,**kwargs)
        theta,x = np.transpose(orbit)
        orbits[i,0] = np.mod(theta,2*np.pi)
        orbits[i,1] = x
        symline_points[i] = x[0]
        residues[i] = get_residue(orbit,mapfn)
        actions[i] = np.sum(mapfn.action(orbits[i]))
        i+=1
    return orbits, residues, actions, symline_points

def get_M_matrix(orbit,cmap):
    M = np.eye(2)
    for pt in orbit:
        M = np.matmul(cmap.jac(pt) , M)
    return M

def get_residue(orbit,mapfn):
    M = get_M_matrix(orbit,mapfn)
    R = 0.25 * (2 - np.trace(M))
    return R