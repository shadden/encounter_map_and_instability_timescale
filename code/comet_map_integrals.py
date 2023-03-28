from celmech.maps import comet_map_ck
import numpy as np
from scipy.integrate import quad

if __name__=="__main__":
    k=3
    q = 1.2
    beta = 1/q
    ck1 = comet_map_ck(k,q)
    print("celmech: {}".format(ck1))
