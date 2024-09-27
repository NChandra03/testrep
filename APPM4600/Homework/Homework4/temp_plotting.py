# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:13:06 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define the parameters
T_i = 20  # Initial soil temperature [degrees C]
T_s = -15  # Constant surface temperature [degrees C]
alpha = 0.138e-6  # Thermal diffusivity [meters^2 per second]
t = 60 * 24 * 60 * 60  # Time in seconds (60 days)

# Define the function T(x)
def T(x):
    return T_s + (T_i - T_s) * erf(x / (2 * np.sqrt(alpha * t)))

# Create an array of x values from 0 to 1
x = np.linspace(0, 1, 100)

# Calculate the corresponding T(x) values
T_values = T(x)

# Plot T(x) against x
plt.figure(figsize=(8, 6))
plt.plot(x, T_values, label='$T(x)$')
plt.title('Temperature Distribution $T(x)$ from 0 to 1 meter')
plt.xlabel('Depth $x$ (meters)')
plt.ylabel('Temperature $T(x)$ (degrees C)')
plt.grid(True)
plt.legend()
plt.show()
