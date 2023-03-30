# MEGNO_Grid
from MEGNO_GridUtils import *
import numpy as np
import sys
I = int(sys.argv[1])
NPROC = int(sys.argv[2])

# Set grid
amin,amax = 150,1e3
qmin,qmax = 30,65
Ncoarse = 3
Nfine = 200
a_edges = np.geomspace(amin,amax,Ncoarse + 1)
q_edges = np.linspace(qmin,qmax,Ncoarse + 1)
a_edges = list(zip(a_edges[:-1],a_edges[1:]))
q_edges = list(zip(q_edges[:-1],q_edges[1:]))
edges = []
for q_edge in q_edges:
    for a_edge in a_edges:
        edges.append((a_edge,q_edge))
arange,qrange = edges[I]

# make grid
par_a = np.geomspace(*arange,Nfine,endpoint=False)
par_q = np.linspace(*qrange,Nfine,endpoint=False)
parameters = []
for q in par_q:
    for a in par_a:
        parameters.append((a,q))

# Run
from rebound.interruptible_pool import InterruptiblePool
pool = InterruptiblePool(processes=NPROC)
results = pool.map(MEGNO_from_aq,parameters)
results2d = np.array(results).reshape(Nfine,Nfine,-1)
print("shape of results: {}".format(results2d.shape))
np.save("./megno_grids/result{}".format(I),results2d)
np.save("./megno_grids/a_range{}".format(I),par_a)
np.save("./megno_grids/q_range{}".format(I),par_q)
