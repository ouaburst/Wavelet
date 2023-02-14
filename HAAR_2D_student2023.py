# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:55:27 2023

@author: Oualid Burström
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot
image = Image.open('data_uppgift1.jpg').convert('LA') 
data = np.copy(np.asarray(image))
datasize=4096
data=data[0:datasize, 0:datasize, 0]
data=data.astype('uint8')
Xfull=data
X=Xfull[0:datasize,0:datasize]

# --- Haar matrix ---

def haar_matrix(n):
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

#---------------------
# --- First sweep ---
#---------------------

h0 = haar_matrix(4096)
s0 = data

# Horizontal matrix multiplication
print("----- horizontal_matrix_mult -----")
horizontal_matrix_mult = np.matmul(s0, h0)
print(horizontal_matrix_mult)

# Display intermediate result
pyplot.imshow(horizontal_matrix_mult, cmap=pyplot.get_cmap('gray'))

# Vertical matrix multiplication
print("----- vertical_matrix_mult -----")
vertical_matrix_mult = np.matmul(np.transpose(h0), horizontal_matrix_mult)
print(vertical_matrix_mult)

# Display final result
pyplot.imshow(vertical_matrix_mult, cmap=pyplot.get_cmap('gray'))

#---------------------
# --- Second sweep ---
#---------------------

compressed1 = np.copy(vertical_matrix_mult)
h1 = haar_matrix(4096)

# Horizontal matrix multiplication
print("----- horizontal_matrix_mult1 -----")
horizontal_matrix_mult1 = np.matmul(compressed1, h1)
print(horizontal_matrix_mult1)

# Display intermediate result
pyplot.imshow(horizontal_matrix_mult1, cmap=pyplot.get_cmap('gray'))

# Vertical matrix multiplication
print("----- vertical_matrix_mult1 -----")
vertical_matrix_mult1 = np.matmul(np.transpose(h1), horizontal_matrix_mult1)
print(vertical_matrix_mult1)

# Display final result
pyplot.imshow(vertical_matrix_mult1, cmap=pyplot.get_cmap('gray'))

'''
#------------------------------------------------------
# Inverse Haar wavelet transform (second sweep)
# Vertical matrix multiplication (inverse)
# h1.(s2')
#------------------------------------------------------

print("----- inverse_vertical_matrix_mult1 -----")
inverse_vertical_matrix_mult1 = np.matmul(h1, vertical_matrix_mult1)
print(inverse_vertical_matrix_mult1)

# Horizontal matrix multiplication (inverse)
# (s1')h1

print("----- inverse_horizontal_matrix_mult1 -----")
inverse_horizontal_matrix_mult1 = np.matmul(inverse_vertical_matrix_mult1, np.transpose(h1))
print(inverse_horizontal_matrix_mult1)

# Display the resulting image
pyplot.imshow(inverse_horizontal_matrix_mult1, cmap=pyplot.get_cmap('gray'))

#-----------------------------------------------------
# Inverse Haar wavelet transform (first sweep) 
# Vertical matrix multiplication (inverse)
# h0.(s1')
#------------------------------------------------------

print("----- inverse_vertical_matrix_mult -----")
inverse_vertical_matrix_mult = np.matmul(h0, vertical_matrix_mult)
print(inverse_vertical_matrix_mult)

# Horizontal matrix multiplication (inverse)
# (s0')h0

print("----- inverse_horizontal_matrix_mult -----")
inverse_horizontal_matrix_mult = np.matmul(inverse_vertical_matrix_mult, np.transpose(h0))
print(inverse_horizontal_matrix_mult)

# Display the resulting image
pyplot.imshow(inverse_horizontal_matrix_mult, cmap=pyplot.get_cmap('gray'))
'''


# --- Thresholding ---
C = 1.0  # Set the threshold value

# Threshold the high-frequency coefficients
thresholded_coeffs = np.copy(vertical_matrix_mult1)
thresholded_coeffs[np.abs(thresholded_coeffs) < C] = 0

# Inverse Haar wavelet transform with thresholding
# Vertical matrix multiplication (inverse)
# h1.(s2')

print("----- inverse_vertical_matrix_mult1 -----")
inverse_vertical_matrix_mult1 = np.matmul(h1, thresholded_coeffs)
print(inverse_vertical_matrix_mult1)

# Horizontal matrix multiplication (inverse)
# (s1')h1

print("----- inverse_horizontal_matrix_mult1 -----")
inverse_horizontal_matrix_mult1 = np.matmul(inverse_vertical_matrix_mult1, np.transpose(h1))
print(inverse_horizontal_matrix_mult1)

# Display the resulting image
pyplot.imshow(inverse_horizontal_matrix_mult1, cmap=pyplot.get_cmap('gray'))

