# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:44:28 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

def power_method(A, tol=1e-6, Nmax=100):
    n = A.shape[0]
    
    # Random initial vector
    q = np.random.rand(n)
    q = q / np.linalg.norm(q)  # Normalize
    
    lam = []  # List to store eigenvalue approximations
    
    for _ in range(Nmax):
        z = A @ q  # Matrix-vector multiplication
        q = z / np.linalg.norm(z)  # Normalize to get the next q
        current_lam = q.T @ A @ q  # Rayleigh quotient to approximate eigenvalue
        lam.append(current_lam)
        
        # Check convergence
        if len(lam) > 1 and abs(lam[-1] - lam[-2]) < tol:
            break
    
    return lam, q


# Plotting results for multiple Hilbert matrices
plt.close('all')
tol = 1e-12  # Convergence tolerance
Nmax = 100  # Maximum number of iterations
A = np.array([[0,1],[1,0]])
#A = np.array([[3,0],[0,2]])
#P = np.array([[2,0],[0,3]])
#P = np.array([[0.31369855, 0.64933064],[0.96324846, 0.23387582]])
#A =  np.array([[1e16,0],[0,1e-16]])
#A = P @ A @ np.linalg.inv(P)
print(A)
for i in range (10):
    lam, q = power_method(A, tol, Nmax)
    iterations = range(1, len(lam) + 1)  # Iteration numbers
    
    # Plot eigenvalue approximations
    lam = np.array(lam)
    plt.plot(iterations, lam)

# Compute all eigenvalues
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue Approximation")
plt.title("Power Method Convergence")
#plt.legend()
plt.grid()
plt.show()