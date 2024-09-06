# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:54:15 2024

@author: Zara Chandra
"""

# Small values case
x1_small = 0.08
x2_small = 0.07
delta_x1 = 1e-8
delta_x2 = -1e-8

# Large values case
x1_large = 1e6
x2_large = 1e6 - 1000

# Exact values
y_small = x1_small - x2_small
y_large = x1_large - x2_large

# Approximate values
y_tilde_small = (x1_small + delta_x1) - (x2_small + delta_x2)
y_tilde_large = (x1_large + delta_x1) - (x2_large + delta_x2)

# Errors
delta_y_small = abs(y_tilde_small - y_small)
delta_y_large = abs(y_tilde_large - y_large)

# Relative Errors
relative_error_small = delta_y_small / abs(y_small)
relative_error_large = delta_y_large / abs(y_large)

print(f"Small values case - Absolute Error: {delta_y_small}, Relative Error: {relative_error_small}")
print(f"Large values case - Absolute Error: {delta_y_large}, Relative Error: {relative_error_large}")
