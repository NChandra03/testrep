# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:11:02 2024

@author: Zara Chandra
"""

import numpy as np

# Define the values for Δb1 and Δb2
delta_b1 = 1e-5 * 1/2# specify the value for Δb1
delta_b2 = 1e-5 * -1/2# specify the value for Δb2

# Calculate Δx1 and Δx2
delta_x1 = delta_b1 + delta_b2 * 10**10 - delta_b1 * 10**10
delta_x2 = delta_b1 + delta_b1 * 10**10 - delta_b2 * 10**10

# Store the results in a numpy array
delta_x = np.array([delta_x1, delta_x2])

print("Δx =", delta_x)