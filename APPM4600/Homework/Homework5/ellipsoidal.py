# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:48:21 2024

@author: Zara Chandra
"""

import numpy as np

# Define the ellipsoid function f(x, y, z)
def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16

# Define the partial derivatives of f with respect to x, y, and z
def fx(x, y, z):
    return 2*x

def fy(x, y, z):
    return 8*y

def fz(x, y, z):
    return 8*z

# Define the iteration function
def iterate_to_ellipsoid(x0, y0, z0, tol=1e-6, max_iter=100):
    x, y, z = x0, y0, z0
    for i in range(max_iter):
        # Compute the function value and partial derivatives at the current point
        f_val = f(x, y, z)
        fx_val = fx(x, y, z)
        fy_val = fy(x, y, z)
        fz_val = fz(x, y, z)
        
        # Compute the denominator f_x^2 + f_y^2 + f_z^2
        denom = fx_val**2 + fy_val**2 + fz_val**2
        
        # If the denominator is zero, stop the iteration
        if denom == 0:
            print("Zero gradient encountered. Stopping iteration.")
            break
        
        # Compute the step size d
        d = f_val / denom
        
        # Update x, y, and z using the iteration scheme
        x_new = x - d * fx_val
        y_new = y - d * fy_val
        z_new = z - d * fz_val
        
        # Compute the error (Euclidean distance between current and new points)
        error = np.sqrt((x_new - x)**2 + (y_new - y)**2 + (z_new - z)**2)
        
        # Print the error at this iteration
        print(f"Iteration {i+1}: Error = {error}")
        
        # Check for convergence
        if error < tol:
            print(f"Converged after {i+1} iterations.")
            return x_new, y_new, z_new
        
        # Update the current values
        x, y, z = x_new, y_new, z_new
    
    print("Max iterations reached without convergence.")
    return x, y, z

# Initial guess
x0, y0, z0 = 1, 1, 1

# Run the iteration to find the point on the ellipsoid
x_sol, y_sol, z_sol = iterate_to_ellipsoid(x0, y0, z0)

print(f"Point on the ellipsoid: x = {x_sol}, y = {y_sol}, z = {z_sol}")

