# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:51:14 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(0,9,10)
#y = np.arange(10)
#print('The first 3 values of x are', x[0:3])

w = 10**(-np.linspace(1,10,10))
x = np.linspace(1, len(w), len(w))
plt.semilogy(x,w)
s = 3 * w
plt.semilogy(x,s)