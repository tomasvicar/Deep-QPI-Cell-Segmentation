import numpy as np
import skfmm

def geodesic_dt(seeds,mask):
    phi = seeds.copy()

    phi = phi.astype(np.float)
    
    phi[phi==0]=-1
    
    phi = -phi
    
    phi  = np.ma.MaskedArray(phi, mask==0)
    
    dt  = skfmm.distance(phi)
    
    dt[seeds] = 0
    
    dt[mask==0]=np.inf
    
    dt = np.array(dt)
    
    return dt