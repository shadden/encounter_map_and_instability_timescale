# MEGNO_Grid
from MEGNO_GridUtils import *
import numpy as np

# Set grid
amin,amax = 150,1e3
qmin,qmax = 30,65

# make grid
Ngrid = 4
par_a = np.geomspace(150,1000,Ngrid)
par_q = np.linspace(30.,65.,Ngrid)
parameters = []
for q in par_q:
    for a in par_a:
        parameters.append((a,q))        

# Run
from rebound.interruptible_pool import InterruptiblePool
pool = InterruptiblePool()
results = pool.map(MEGNO_from_aq,parameters)

results2d = np.array(results).reshape(Ngrid,Ngrid,-1)
print("shape of results: {}".format(results2d.shape))