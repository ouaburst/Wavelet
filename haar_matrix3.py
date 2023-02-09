# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:11:09 2023

@author: Budokan
"""

import numpy as np


def haar(width):
    h = np.zeros(width)
    h[:width//2] = 1
    h[width//2:] = -1
    return h
    
print("haar matrix \n", haar(8))    