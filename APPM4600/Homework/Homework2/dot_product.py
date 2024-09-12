# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:31:56 2024

@author: Zara Chandra
"""

import numpy as np

# Create vector t with entries starting at 0, incrementing by π/30 to π
t = np.linspace(0, np.pi, 31)

# Create vector y = cos(t)
y = np.cos(t)

def dot(t,y):
    total = 0
    for i in range (len(t)):
        total += t[i] * y[i]
    return total

test = dot(t,y)
test1 = np.dot(t,y)

print('The sum is ', test)
