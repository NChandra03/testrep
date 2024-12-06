# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:04:32 2024

@author: Zara Chandra
"""

from mpmath import mp, mpf, matrix, norm, sqrt, nstr

# Set the precision
mp.dps = 50  # Decimal places of precision

def hilbert_matrix_mp(n):
    """
    Generate the Hilbert matrix of size n x n with high precision.
    """
    return matrix([[mpf(1) / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])


def power_method_mp(A, tol=mpf("1e-30"), max_iter=1000):
    """
    Power Method to find the dominant eigenvalue and eigenvector with high precision.

    Parameters:
    A (mpmatrix): Square matrix with high precision.
    tol (mpf): Convergence tolerance.
    max_iter (int): Maximum number of iterations.

    Returns:
    lam (mpf): Approximation of the dominant eigenvalue.
    v (matrix): Approximation of the corresponding eigenvector.
    iter_count (int): Number of iterations needed for convergence.
    """
    n = A.rows
    # Generate a random initial vector with high precision
    q = matrix([mpf(mp.rand()) for _ in range(n)])  
    q /= norm(q)  # Normalize
    lam_old = mpf("0")

    for i in range(max_iter):
        z = A @ q
        q = z / norm(z)  # Normalize
        lam = (q.T @ A @ q)[0]  # Rayleigh quotient
        if abs(lam - lam_old) < tol:  # Convergence check
            return lam, q, i + 1
        lam_old = lam

    raise Exception("Power method did not converge within the maximum number of iterations")


def inverse_power_method_mp(A, tol=mpf("1e-30"), max_iter=1000):
    """
    Inverse Power Method to find the smallest eigenvalue of a matrix with high precision.

    Parameters:
    A (mpmatrix): Square matrix with high precision.
    tol (mpf): Convergence tolerance.
    max_iter (int): Maximum number of iterations.

    Returns:
    lam (mpf): Approximation of the smallest eigenvalue.
    v (matrix): Approximation of the corresponding eigenvector.
    iter_count (int): Number of iterations needed for convergence.
    """
    n = A.rows
    q = matrix([mpf(mp.rand()) for _ in range(n)])  # Random initial vector
    q /= norm(q)  # Normalize
    lam_old = mpf("0")

    for i in range(max_iter):
        z = matrix(mp.lu_solve(A, q))  # Solve the system A * x = q
        q = z / norm(z)  # Normalize
        #lam_reciprocal = (q.T @ A @ q)[0]  # Rayleigh quotient (this is 1 / lambda)
        lam = (q.T @ A @ q)[0]
        # Compute the smallest eigenvalue as the reciprocal
        #lam = 1 / lam_reciprocal if lam_reciprocal != 0 else mpf("inf")
        
        # Check for convergence
        if abs(lam - lam_old) < tol:
            return lam, q, i + 1
        lam_old = lam

    raise Exception("Inverse power method did not converge within the maximum number of iterations")


# Part (a): Find the dominant eigenvalue for n = 4, 8, 12, 16, 20
print("Part (a): Dominant Eigenvalue")
for n in range(4, 21, 4):
    H = hilbert_matrix_mp(n)
    dominant_eigenvalue, eigenvector, iterations = power_method_mp(H)
    print(f"n = {n}: Dominant Eigenvalue = {nstr(dominant_eigenvalue, 20)}, Iterations = {iterations}")

# Part (b): Find the smallest eigenvalue for n = 16
print("\nPart (b): Smallest Eigenvalue")
n = 16
H = hilbert_matrix_mp(n)
smallest_eigenvalue, eigenvector, iterations = inverse_power_method_mp(H)
print(f"n = {n}: Smallest Eigenvalue = {nstr(smallest_eigenvalue, 20)}, Iterations = {iterations}")
