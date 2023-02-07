import numpy as np
import pywt
import pywt.data
from PIL import Image

def Wavelet_matrix(n):
    h = np.array([1, 1]) / np.sqrt(2)
    H = np.kron(h, np.eye(n))
    return H

def haar_wavelet_transform(data, n, block_size=512):
    H = Wavelet_matrix(block_size)
    data = data.reshape(-1, block_size)
    transformed_data = np.empty((data.shape[0], block_size))
    for i, block in enumerate(data):
        transformed_block = np.dot(H, block)
        transformed_data[i] = transformed_block
    return transformed_data

# Load image
data = np.copy(np.asarray(Image.open('data_uppgift1.jpg').convert('LA')))
datasize = 4096
data = data[0:datasize, 0:datasize, 0]
data = data.astype('uint8')

# Perform wavelet transform
n = data.shape[0]
transformed_data = haar_wavelet_transform(data, n)