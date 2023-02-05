# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:25:08 2023

@author: MNBE
"""
# fil för datainläsning 

import matplotlib.pyplot as plt
import numpy as np
#import cupy as cp
import os
from mpl_toolkits.mplot3d import Axes3D
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

def Wavelet_matrix(data, wavelet_type='haar'):
    """
    Calculates h0 and s0 using the 2D discrete wavelet transform (DWT).
    
    data: 2D matrix containing the image data
    wavelet_type: type of wavelet to be used for the DWT (default is 'db1')
    
    Returns: h0, s0 (the horizontal and vertical detail coefficients)
    """
    # Perform 2D DWT on the image data
    coeffs = pywt.dwt2(data, wavelet_type)

    # Get the horizontal and vertical detail coefficients (h0 and s0)
    cA, (cH, cV, cD) = coeffs
    
    # h0 is the horizontal detail coefficients
    h0 = cH

    # s0 is the vertical detail coefficients
    s0 = cV
    
    return h0, s0

def matrix_multiplication(A, B):
    return np.matmul(A, B)

# Calculate h0 and s0 using the Wavelet_matrix function
h0, s0 = Wavelet_matrix(data,'haar')

print ('h0', h0)
print ('s0', s0)

# Matrix multiplication horizontally

print(matrix_multiplication(s0, h0))