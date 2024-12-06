# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 01:26:45 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def power_method(A, Nmax):
    """
    Power Method to find the dominant eigenvalue and eigenvector.

    Parameters:
    A (ndarray): Square matrix.
    Nmax (int): Maximum number of iterations.

    Returns:
    lam (list): Approximations of the dominant eigenvalue at each iteration.
    v (ndarray): Approximation of the dominant eigenvector.
    """
    n = A.shape[0]
    
    # Random initial vector
    q = np.random.rand(n)
    q = q / np.linalg.norm(q)  # Normalize
    
    lam = []  # List to store eigenvalue approximations
    
    for _ in range(Nmax):
        z = A @ q  # Matrix-vector multiplication
        q = z / np.linalg.norm(z)  # Normalize to get the next q
        lam.append(q.T @ A @ q)  # Rayleigh quotient to approximate eigenvalue
    
    return lam, q


# Main test function
def main():
    # Generate a 2000x2000 random matrix
    n = 2000
    A = np.random.randn(n, n)

    # Create diagonal eigenvalues
    dd = np.ones(n)
    dd[0] = 40  # Set the largest eigenvalue to 40
    
    # Make matrix A have eigenvalues specified in dd
    A = A @ np.diag(dd) @ np.linalg.inv(A)
    
    # Compute eigenvalues and eigenvectors with numpy's eig function
    start = time.time()
    _, eigvals = np.linalg.eig(A)
    end = time.time()
    print(f"Runtime for numpy eig: {end - start:.4f} seconds")
    
    # Run the Power Method
    Nmax = 100
    start = time.time()
    lam, v = power_method(A, Nmax)
    end = time.time()
    print(f"Runtime for Power Method: {end - start:.4f} seconds")
    
    # Compute error compared to the true dominant eigenvalue
    err = [abs(l - max(dd)) for l in lam]
    
    # Plot error convergence
    N = np.arange(1, Nmax + 1)
    theoretical_decay = (1 / dd[0]) ** N
    plt.semilogy(N, err, 'o-', label="Power Method Error")
    plt.semilogy(N, theoretical_decay, 'r', label="Theoretical Decay")
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.legend()
    plt.title("Power Method Convergence")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
