# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:37:20 2024

@author: Zara Chandra
"""
import numpy as np
import matplotlib.pyplot as plt
import random

plt.close('all')
# Parameters
R = 1.2
delta_r = 0.1
f = 15
p = 0

# Theta values from 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 1000)

# Calculate x(θ) and y(θ)
x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)

plt.figure(1)
plt.plot(x,y)
plt.axis('equal')

plt.figure(2)
plt.axis('equal')
for i in range (10):
    R = i
    delta_r = 0.05
    f = 2 + i
    p = random.uniform(0,2)
    # Calculate x(θ) and y(θ)
    x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
    y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(x,y)
    
    