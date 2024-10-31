# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:03:15 2024

@author: Zara Chandra
"""

import numpy as np

def eval_legendre(n,x):
    L = np.zeros(n+1)
    L[0] = 1
    L[1] = x
    for i in range(1,n):
        L[i + 1] = (1/(i + 1)) * ((2 * i + 1) * x * L[i] - i * L[i-1])
    return(L)
    
print(eval_legendre(4, 2))