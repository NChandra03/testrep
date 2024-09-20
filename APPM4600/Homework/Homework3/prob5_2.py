# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:19:28 2024

@author: Zara Chandra
"""

import math

def fixed_point_iteration(x0, tol=1e-10, max_iter=1000):
    def g(x):
        return -math.sin(2*x) + (5*x)/4 - 3/4
    
    x_n = x0
    for i in range(max_iter):
        x_next = g(x_n)
        if abs(x_next - x_n) < tol:
            return x_next, i
        x_n = x_next
    return None, max_iter

# Example of usage
initial_guesses = [-1,-0.7,0,1.6,1.8,3,3.2,4.4,4.6]
for guess in initial_guesses:
    root, iterations = fixed_point_iteration(guess)
    if root:
        print(f"Root found: {root:.10f} after {iterations} iterations with initial guess {guess}")
    else:
        print(f"No convergence with initial guess {guess}")
