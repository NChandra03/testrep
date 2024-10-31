# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:49:10 2024

@author: Zara Chandra
"""

import numpy as np

def eval_legendre(n):
    # Array to hold the Legendre polynomial functions
    p_array = []

    # Define the 0th and 1st Legendre polynomials
    p_array.append(lambda x: 1)
    if n > 0:
        p_array.append(lambda x: x)

    # Generate remaining polynomials using the recurrence relation
    for i in range(1, n):
        def legendre_poly(x, i=i):
            return (1 / (i + 1)) * ((2 * i + 1) * x * p_array[i](x) - i * p_array[i - 1](x))
        
        p_array.append(legendre_poly)

    return p_array
    
# Example usage
n = 4
p_array = eval_legendre(n)
x = 2  # example input value

# Evaluate each polynomial at x
results = [p(x) for p in p_array]
print("Legendre polynomial values at x =", x, ":", results)