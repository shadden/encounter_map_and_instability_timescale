from celmech.maps import comet_map_ck
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
def Ck_approx(k,beta):
    E = np.exp(1)
    exp_arg  = -k * 2*np.sqrt(2) / 3 / beta**1.5
    exp_arg *= 1 + beta**(9/4) / 2**(7/4)
    Ck  = 1/np.sqrt(k)
    Ck /= gamma(0.5 - k)
    Ck /= gamma(1+k)
    Ck *= (-1)**k * np.pi * E**k * 2**(11/8 + 0.5 * k)
    Ck *= beta**(0.25 - k/2)
    Ck /= np.sqrt(2**1.25 - beta**0.75)
    Ck *= np.exp(exp_arg)
    return Ck


if __name__=="__main__":
    from matplotlib import pyplot as plt
    N = 20
    qmin,qmax = 1.2,3
    qvals = np.geomspace(qmin,qmax,N)
    for k in range(2,8):
        Cexact = np.zeros(N)
        Capprox = np.zeros(N)
        for i,q in enumerate(qvals):
            Cexact[i] = comet_map_ck(k,q)
            Capprox[i] = Ck_approx(k,1/q)
        plt.plot(qvals,np.abs((Cexact-Capprox)/Cexact),label="numerical")
    #plt.plot(qvals,Cexact,label="numerical")
    #plt.plot(qvals,Capprox,label="approx")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
