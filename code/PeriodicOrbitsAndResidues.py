import numpy as np
from scipy.optimize import root_scalar

def symmetry_line1(s):
    return np.array((0,s))
def symmetry_line2(s):
    return np.array((np.pi,s))
def symmetry_line3(s):
    return np.array((s/2,s))


def find_periodic_orbit(m,n,mapfn,guess=None,minimax=False):
    
    mapfn.mod_p = False
    mapfn.mod_theta = False

    if minimax:
        symline = symmetry_line1
    else:
        symline = symmetry_line2 if n%2 else symmetry_line3
    rootfn = lambda s: nest_map(symline(s),mapfn,n)[0] - symline(s)[0] - 2 * np.pi * m
    if guess is None:
        guess = 2 * m * np.pi / n
    xmin = 2 * (m-1) * np.pi / n
    x0  = 2 * m * np.pi / n
    xmax = 2 * (m+1) * np.pi / n
    
    bracket = [xmin + 0.5*(guess -xmin),guess]
    rt = root_scalar(rootfn,x0=guess,x1 = bracket[0])
    pt0 = symline(rt.root)    
    orbit = nest_map_list(pt0,mapfn,n)
    
    return orbit

