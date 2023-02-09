# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 22:41:37 2023

@author: Budokan
"""

import numpy as np

def haarmatrix(N):
    n = np.floor(np.log(N)/np.log(2))

    if 2**n != N: raise Exception('error: size '+str(N)+' is not multiple of power of 2')

    z = np.resize(1.*np.arange(N)/N, (len(1.*np.arange(N)), len(1.*np.arange(N)/N)))
    k = np.transpose(np.resize(1.*np.arange(N), (len(1.*np.arange(N)/N), len(1.*np.arange(N)))))
    
    
    p  = np.floor(np.log(np.maximum(1,k))/np.log(2))
    q  = k - (2**p) + 1
    z1 = (q-1)   / (2**p)
    z2 = (q-0.5) / (2**p)
    z3 = q       / (2**p)
    A  = (1/np.sqrt(N)) * ((( 2**(p/2.)) * ((z >= z1) & (z < z2))) + ((-2**(p/2.)) * ((z >= z2) & (z < z3))))
    A[0,:] = 1/np.sqrt(N)
    return A

print("haar matrix \n", haarmatrix(4))