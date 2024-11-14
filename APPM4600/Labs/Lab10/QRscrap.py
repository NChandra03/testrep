# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:52:08 2024

@author: Zara Chandra
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad
def driver():
    plt.close('all')
    # function you want to approximate
    #f = lambda x: math.exp(x)
    f = lambda x: x * np.exp(-x ** 2)
    # Interval of interest
    a = -3
    b = 3
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 5
    # Number of points you want to sample in [a,b]
    N = 10
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)
    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
        #''' create vector with exact values'''
        fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
    plt.figure(1)
    plt.plot(xeval,fex, label= 'f(x)')
    plt.plot(xeval,pval,label= 'Expansion')
    plt.legend()
    plt.show()
    err = abs(pval-fex)
    plt.figure(2)
    plt.semilogy(xeval,err,label='error')
    plt.legend()
    plt.show()


def eval_legendre_expansion(f, a, b, w, n, x):
    # Get the Legendre polynomials up to degree n
    p = eval_legendre(n)
    
    # Initialize the sum to 0
    pval = 0.0
    
    # Loop over the polynomials
    for j in range(n + 1):
        # Define numerator and denominator functions for integration
        numerator = lambda x_val: p[j](x_val) * f(x_val) * w(x_val)
        denominator = lambda x_val: (p[j](x_val) ** 2) * w(x_val)
        
        # Perform the integrations
        num_quad, _ = quad(numerator, a, b)
        denom_quad, _ = quad(denominator, a, b)
        
        # Calculate the coefficient aj
        aj = num_quad / denom_quad
        
        # Accumulate the Legendre expansion value
        pval += aj * p[j](x)
    
    return pval

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

if __name__ == '__main__':
# run the drivers only if this is called from the command line
    driver()