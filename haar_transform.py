# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:40:23 2023

@author: Budokan
"""

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
pyplot.imshow(X, cmap=pyplot.get_cmap('gray'))

def haar_transform(data):
    data = np.asarray(data)
    rows, cols = data.shape
    if rows != cols:
        raise ValueError("Input matrix must be square")
    if np.log2(rows) % 1 != 0:
        raise ValueError("Number of rows must be power of 2")
    n = rows
    haar_matrix = np.zeros((n, n))
    sqrt_2 = np.sqrt(2)
    for i in range(n):
        for j in range(n):
            if j >= n // 2:
                haar_matrix[i, j] = sqrt_2 / 2
            else:
                haar_matrix[i, j] = -sqrt_2 / 2
    transformed = np.dot(np.dot(haar_matrix, data), haar_matrix.T)
    return transformed




print("Input Matrix:")
print(data)

transformed = haar_transform(data)
print("Haar Transform of the Input Matrix:")
print(transformed)

pyplot.imshow(transformed, cmap=pyplot.get_cmap('gray'))