# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:33:23 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x - 4 * np.sin(2 * x) - 3

# Generate x values from -2pi to 2pi
x = np.linspace(-2 * np.pi, 4 * np.pi, 1000)
# Calculate y values
y = f(x)

# Plotting the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = x - 4\sin(2x) - 3$', color='b')
plt.title('Plot of $f(x) = x - 4\sin(2x) - 3$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()
