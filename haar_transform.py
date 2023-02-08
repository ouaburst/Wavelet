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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def haar_matrix(n):
    haar = np.zeros((n, n))
    i_start = 0
    for i in range(0, n, 2):
        haar[i, i_start:i_start + n // 2] = 1
        haar[i + 1, i_start:i_start + n // 2] = -1
        i_start += n // 2
    return haar

def haar_transform(x):
    n = len(x)
    H = haar_matrix(n)
    return H @ x

def haar2d(img_array):
    H = haar_matrix(img_array.shape[0])
    img_array = H @ img_array @ H.T
    H = haar_matrix(img_array.shape[1])
    img_array = img_array @ H
    img_array = img_array.T
    img_array = img_array @ H.T
    return img_array

# Load the image
image = Image.open('data_uppgift1.jpg')

data = np.copy(np.asarray(image))
datasize=4096
data=data[0:datasize, 0:datasize, 0]
data=data.astype('uint8')

# Convert the image to a numpy array
#img_array = np.array(data)

# Apply the 2D Haar transform
result = haar2d(img_array)

# Flatten the image array
#img_flat = img_array.flatten()

# Apply the Haar transform
result = haar_transform(img_flat)

# Plot the new image
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
