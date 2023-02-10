# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:32:40 2023

@author: Oualid Burström
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 22:41:37 2023

@author: Budokan
"""

import numpy as np

def Haar_matrix(n):
    m = 0
    if n % 2 != 0:
        raise ValueError("n must be an even number")
    matrix = np.zeros((n, n))
    for i in range(n // 2):            
        matrix[m, i] = 0.5        
        matrix[m+1, i] = 0.5
        matrix[m, i + n // 2] = 0.5        
        matrix[m+1, i + n // 2] = -0.5
        m += 2
    return matrix

print("Haar matrix \n", Haar_matrix(4096))