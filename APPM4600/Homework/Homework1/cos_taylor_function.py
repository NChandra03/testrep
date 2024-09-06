# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 01:19:24 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# Define the values for delta
delta_values = np.logspace(-16, 0, num=100)

# Define the original function
def lhs(x, delta):
    return np.cos(x + delta) - np.cos(x)

# Define the trigonometric identity (rhs)
def rhs(x, delta):
    return -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

# Define the Taylor series approximation for the original expression
def taylor(x, delta):
    epsilon = x + delta/2
    return -delta * np.sin(x) - (delta**2 / 2) * np.cos(epsilon)

# Values of x for the two plots
x_values = [np.pi, 10**6]

# Create plots for each x value
for x in x_values:
    lhs_values = lhs(x, delta_values)
    rhs_values = rhs(x, delta_values)
    taylor_values = taylor(x, delta_values)
    
    # Plot all three comparisons on the same graph
    plt.figure()
    plt.plot(delta_values, lhs_values - rhs_values, label="LHS - RHS")
    plt.plot(delta_values, lhs_values - taylor_values, label="LHS - Taylor", linestyle='dashed')
    plt.plot(delta_values, rhs_values - taylor_values, label="RHS - Taylor", linestyle='dotted')
    plt.xscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Difference')
    plt.title(f'Comparisons for x = {x}')
    plt.legend()
    plt.grid(True)
    plt.show()


