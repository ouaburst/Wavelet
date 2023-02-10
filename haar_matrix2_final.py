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

import numpy as np

def Haar_matrix(n):
    # Initialize row counter
    m = 0
    # Check if n is even
    if n % 2 != 0:
        raise ValueError("n must be an even number")
    # Create a zero matrix of size n x n using NumPy
    matrix = np.zeros((n, n))
    # Loop over the first half of the rows
    for i in range(n // 2):            
        # Set values in the matrix according to the Haar matrix formula
        matrix[m, i] = 0.5        
        matrix[m+1, i] = 0.5
        matrix[m, i + n // 2] = 0.5        
        matrix[m+1, i + n // 2] = -0.5
        # Increment the row counter by 2
        m += 2
    # Return the resulting matrix
    return matrix

print("Haar matrix \n", Haar_matrix(4096))