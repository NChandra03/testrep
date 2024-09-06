# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:03:45 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x
x = np.arange(1.920, 2.081, 0.001)

# Part i: Evaluating p(x) using its coefficients
coefficients = [1, -18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512]
p_coefficients = np.polyval(coefficients, x)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(x, p_coefficients, label=r'$p(x)$ via coefficients')
plt.title('Plot of $p(x)$ via coefficients')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.grid(True)
plt.legend()
plt.show()

# Part ii: Evaluating p(x) via the expression (x - 2)^9
p_expression = (x - 2)**9

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(x, p_expression, label=r'$p(x)$ via $(x - 2)^9$')
plt.title('Plot of $p(x)$ via $(x - 2)^9$')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.grid(True)
plt.legend()
plt.show()
