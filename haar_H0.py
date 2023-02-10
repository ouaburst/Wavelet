# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:06:09 2023

@author: Budokan
"""

import numpy as np

def haarMatrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

def haarTransformHorizontal(matrix):
    n, m = matrix.shape
    if m == 1:
        return matrix
    H = haarMatrix(m)
    return np.dot(matrix, H)

# Define the original 4x4 matrix
original_matrix = np.array([[9, 7, 6, 2], [5, 3, 4, 4], [8, 2, 4, 0], [6, 0, 2, 2]])

# Perform the horizontal Haar transform
horizontal_haar = haarTransformHorizontal(original_matrix)

#print("Horizontal Haar transform of the original matrix:\n", horizontal_haar)





matrix1 = np.array([[9, 7, 6, 2], [5, 3, 4, 4], [8, 2, 4, 0], [6, 0, 2, 2]])
matrix2 = np.array([[0.5, 0, 0.5, 0], [0.5, 0, -0.5, 0], [0, 0.5, 0, 0.5], [0, 0.5, 0, -0.5]])
result2 = np.dot(matrix1, matrix2)
#print("result2\n", result2)

print(haarMatrix(6,normalized=False))