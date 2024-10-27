# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:39:37 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def driver():
    plt.close('all')
    
    # Define the function we want to interpolate
    f = lambda x: 1/(1 + (10 * x) ** 2)
    
    # Number of interpolation nodes
    N = 28
    a = -1
    b = 1
    
    # Create interpolation nodes using Chebyshev points
    h = 2 / (N - 1)
    xint = np.array([-1 + (j - 1) * h for j in range(1, N + 1)])
    #xint = np.array([np.cos((2 * j - 1) * np.pi / (2 * N)) for j in range(1, N + 1)])
    print(xint)
    
    # Interpolation data (function values at the interpolation nodes)
    yint = f(xint)
    
    # Precompute barycentric weights
    w = compute_barycentric_weights(xint)
    
    # Create points for evaluating the Lagrange interpolating polynomial
    Neval = 1000
    xeval = np.linspace(a, b, Neval+1)
    yeval_bary = np.zeros(Neval+1)
  
    # Evaluate the barycentric Lagrange interpolation polynomial
    for kk in range(Neval+1):
        yeval_bary[kk] = eval_barycentric(xeval[kk], xint, yint, w)
    
    # Exact values for comparison
    fex = f(xeval)
       
    # Plotting the function and the interpolation
    plt.figure()    
    plt.plot(xeval, fex, label='function')
    plt.plot(xeval, yeval_bary, alpha=0.5, label='barycentric Lagrange') 
    plt.plot(xint,yint,'o', label = 'nodes')
    plt.title('Interpolation')
    plt.legend()

    # Plotting the error
    plt.figure() 
    err_bary = abs(yeval_bary - fex)
    print('Error norm:', norm(err_bary))
    plt.semilogy(xeval, err_bary, label='Barycentric Lagrange', alpha=0.5)
    plt.title('Error')
    plt.legend()
    plt.show()
    
    # Plotting the function and the interpolation within the range x = [-0.1, 0.1]
    plt.figure()    
    plt.plot(xeval, fex, label='function')
    plt.plot(xeval, yeval_bary, alpha=0.5, label='barycentric Lagrange') 
    plt.plot(xint, yint, 'o', label='nodes', alpha = 0.5)
    plt.xlim([-0.1, 0.1])  # Set x-axis limits
    plt.title('Interpolation (x = -0.1 to 0.1)')
    plt.legend()
    
    # Plotting the error within the range x = [-0.1, 0.1]
    plt.figure() 
    err_bary = abs(yeval_bary - fex)
    print('Error norm:', norm(err_bary))
    plt.semilogy(xeval, err_bary, label='Barycentric Lagrange', alpha=0.5)
    plt.xlim([-0.1, 0.1])  # Set x-axis limits
    plt.title('Error (x = -0.1 to 0.1)')
    plt.legend()
    
    plt.show()


def compute_barycentric_weights(xint):
    """ Compute the barycentric weights for the interpolation nodes """
    N = len(xint)
    w = np.ones(N)
    
    for j in range(N):
        for i in range(N):
            if i != j:
                w[j] /= (xint[j] - xint[i])
    
    return w

def eval_barycentric(x, xint, yint, w):
    """ Evaluate the barycentric Lagrange polynomial at a point x """
    N = len(xint)
    
    numerator = 0
    denominator = 0
    
    for j in range(N):
        if x == xint[j]:
            return yint[j]
        term = w[j] / (x - xint[j])
        numerator += term * yint[j]
        denominator += term
    
    return numerator / denominator

# Run the driver function
driver()
