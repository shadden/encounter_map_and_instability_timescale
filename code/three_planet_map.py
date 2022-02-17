import numpy as np
from celmech.miscellaneous import sk

def generate_jk_pairs(J,kmax):
    """
    Generate list of rational numbers j/(j-k)
    between (2*J-1)/(2*(J-1)-1) and (2*J+1)/(2*J-1) 
    up to maximum value of k=kmax.
    Fractions are not reduced.
    """
    jk_pairs = []
    for k in range(1,kmax+1):
        jmin = int(np.ceil((2 * J - 1) * k / 2))
        jmax = int(np.floor((2 * J + 1) * k / 2))
        for j in range(jmin,jmax+1):
            jk_pairs.append((j,k))
    return jk_pairs
def generate_jk_pairs(J,kmax):
    """
    Generate list of rational numbers j/(j-k)
    between (2*J-1)/(2*(J-1)-1) and (2*J+1)/(2*J-1) 
    up to maximum value of k=kmax.
    Fractions are not reduced.
    """
    jk_pairs = []
    for k in range(1,kmax+1):
        jmin = int(np.ceil((2 * J - 1) * k / 2))
        jmax = int(np.floor((2 * J + 1) * k / 2))
        for j in range(jmin,jmax+1):
            jk_pairs.append((j,k))
    return jk_pairs

class ThreePlanetMap():
    def __init__(self,eps1,eps2,J1,J2,etp,n1=None,n2=None,kmax = 5):
        self.J1 = J1
        if n1:
            self.n1 = n1 #J1 / (J1-1)
        else:
            self.n1 = J1 / (J1-1)
        self.a1 = self.n1**(-2/3)
        
        self.J2 = J2
        if n2:
            self.n2 = n2 
        else:
            self.n2=J2/(J2+1)
        self.a2 = self.n2**(-2/3)
        
        self.da0_1 = 1-self.a1
        self.da0_2 = self.a2-1
        
        self.eps1=eps1
        self.eps2=eps2
        
        self.etp = etp
        
        self.ecross1 = self.da0_1
        self.ecross2 = self.da0_2 / self.a2
        
        self.etp = etp
        self.y1 = etp/self.ecross1
        self.y2 = etp/self.ecross2
        
        self.kmax = kmax
        self.sk1_vals = np.array([sk(k,self.y1) for k in range(1,kmax+1)])
        self.sk2_vals = np.array([sk(k,self.y2) for k in range(1,kmax+1)])
        
        self.jk_pairs1 = generate_jk_pairs(J1-1,kmax)
        self.jk_pairs2 = generate_jk_pairs(J2+1,kmax)
                
        self.res_orders = np.array([k for j,k in self.jk_pairs2],dtype=int)

        self.n21 = self.n2 - self.n1
        
        self.amps1 = self.eps1 * np.array([(-1)**(k+1) *self.sk1_vals[k-1] for k in self.res_orders])
        self.amps2 = self.eps2 * np.array([(-1)**k * self.sk2_vals[k-1] for k in self.res_orders])
        
        self.j_times_amps1 = np.array([-1*jk[0]*amp for jk,amp in zip(self.jk_pairs1,self.amps1)])
        self.j_times_amps2 = np.array([jk[0]*amp for jk,amp in zip(self.jk_pairs2,self.amps2)])
        self.k_times_amps1 = np.array([k*amp for k,amp in zip(self.res_orders,self.amps1)])
        self.k_times_amps2 = np.array([k*amp for k,amp in zip(self.res_orders,self.amps2)])
    
    def H0(self,z):
        psi,M,dpsi,pa,pe,L = z
        n1 = self.n1
        return pe + (n1-1) * pa - 0.75 * (pa-pe)*(pa-pe) + self.n21 * L
    
    def H1(self,z):
        psi,M,dpsi,pa,pe,L = z
        angs1 = self.jk_pairs1 @ np.array([-1*psi,M])
        angs2 = self.jk_pairs2 @ np.array([psi + dpsi,M])
        H1 = self.amps1 @ np.cos(angs1)
        H1 += self.amps2 @ np.cos(angs2)
        return H1
    
    def H(self,z):
        return self.H0(z)+self.H1(z)
    
    def H0flow(self,z):
        psi,M,dpsi,pa,pe,L = z
        n1=self.n1
        n21 = self.n21
        dfreq = 1.5 * (pe-pa)
        return np.array([n1-1+dfreq,1-dfreq,n21,0,0,0])
        
    def H1flow(self,z):
        psi,M,dpsi,pa,pe,L = z
        angs1 = self.jk_pairs1 @ np.array([-1*psi,M])
        angs2 = self.jk_pairs2 @ np.array([psi + dpsi,M])

        padot = self.j_times_amps1 @ np.sin(angs1)
        padot += self.j_times_amps2 @ np.sin(angs2)
        
        pedot = self.k_times_amps1 @ np.sin(angs1)
        pedot += self.k_times_amps2 @ np.sin(angs2)
        
        Ldot = self.j_times_amps2 @ np.sin(angs2)
        
        return np.array([0,0,0,padot,pedot,Ldot])
    
    def Hflow(self,z):
        return self.H0flow(z) + self.H1flow(z)
    
    def H0step(self,z,h):
        return z + h * self.H0flow(z)
    def H1step(self,z,h):
        return z + h * self.H1flow(z)
    def DKDstep(self,z,h):
        h_2 = 0.5*h
        z = self.H0step(z,h_2)
        z = self.H1step(z,h)
        z = self.H0step(z,h_2)
        return z
    def integrate(self,z,tnow,tfin,dt):
        Nsteps = int((tfin-tnow)//dt)
        z=self.H0step(z,0.5*dt)
        for _ in range(Nsteps):
            z=self.H1step(z,dt)
            z=self.H0step(z,dt)
        z=self.H0step(z,-0.5*dt)
        return z,tnow + Nsteps*dt        
