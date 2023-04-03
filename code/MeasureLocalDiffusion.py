from celmech.maps import CometMap
import numpy as np
mN = 5.15e-5
N = 20
beta_vals = np.linspace(30/65.,30/35.,5)
cmap = CometMap(mN,20,44./30.)
epscrit = cmap.get_eps_crit()

