# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:25:08 2023

@author: MNBE
"""
# fil för datainläsning 

#import matplotlib.pyplot as plt
import numpy as np
#import cupy as cp
#import os
#from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib import pyplot
#from keras.datasets import mnist
import time
import pywt
import pywt.data

tic = time.time()
# Load image
original = pywt.data.camera()
image = Image.open('data_uppgift1.jpg').convert('LA') 
data = np.copy(np.asarray(image))
datasize=4096
data=data[0:datasize, 0:datasize, 0]
data=data.astype('uint8')
#(train_X, train_y), (test_X, test_y) = mnist.load_data()
#Xfull=train_X[0]
Xfull=data
X=Xfull[0:datasize,0:datasize]
#X=[[9,7,6,2],[5,3,4,4],[8,2,4,0],[6,0,2,2]]
#pyplot.imshow(X, cmap=pyplot.get_cmap('gray'))

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

# --- Compress the image ---

h0 = haar_matrix(4096)
s0 = data

# Horizontal matrix multiplication 
# s0.h0

print("----- horizontal_matrix_mult -----")
horizontal_matrix_mult = np.matmul(s0, h0)
print(horizontal_matrix_mult)

pyplot.imshow(horizontal_matrix_mult, cmap=pyplot.get_cmap('gray'))


# Vertical matrix multiplication
# transpose(h0).(s0.h0)

print("----- vertical_matrix_mult -----")
vertical_matrix_mult = np.matmul(np.transpose(h0), horizontal_matrix_mult)
print(vertical_matrix_mult)

pyplot.imshow(vertical_matrix_mult, cmap=pyplot.get_cmap('gray'))
