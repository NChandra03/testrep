# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:48:02 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

def hilbert_matrix(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])

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
tol = 1e-6  # Convergence tolerance
Nmax = 100  # Maximum number of iterations

#n_list = [4,8,12,16,20]
n_list = [16]
for n in n_list:
    print(f"Matrix size n = {n}")
    #A = hilbert_matrix(n)
    A = np.linalg.inv(hilbert_matrix(n))
    lam, q = power_method(A, tol, Nmax)
    iterations = range(1, len(lam) + 1)  # Iteration numbers
    
    # Plot eigenvalue approximations
    lam = np.array(lam)
    lam = 1/lam
    plt.plot(iterations, lam, label=f"n = {n}")
    
    # Compute all eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)  
    smallest_eigenvalue = np.max(eigenvalues)
    print(1/smallest_eigenvalue)
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue Approximation")
plt.title("Power Method Convergence for Hilbert Matrices")
plt.legend()
plt.grid()
plt.show()
