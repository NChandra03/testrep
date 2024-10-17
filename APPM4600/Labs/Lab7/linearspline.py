# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:35:41 2024

@author: Zara Chandra
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import norm

def driver(): 
    plt.close('all')

    f = lambda x: 1/(1 + (10 * x) ** 2)  # Exact function
    
    N = 20  # Number of interpolation points
    a = -1  # Interval start
    b = 1   # Interval end
    
    ''' Create interpolation nodes'''
    xint = np.linspace(a, b, N)
    yint = f(xint)
    
    ''' New arrays for interpolated points '''
    x_interp = []
    y_interp = []
    
    ''' Evaluate between each consecutive pair of points '''
    for i in range(N-1):
        # Generate 20 points between xint[i] and xint[i+1]
        x_vals = np.linspace(xint[i], xint[i+1], 2)
        for x in x_vals:
            y = lineval(xint[i], yint[i], xint[i+1], yint[i+1], x)
            x_interp.append(x)
            y_interp.append(y)
    
    # Convert to numpy arrays
    x_interp = np.array(x_interp)
    y_interp = np.array(y_interp)
    
    ''' Plot the nodes and the interpolated function '''
    plt.figure()
    plt.plot(xint, yint, 'ro', label='Interpolation Nodes')  # Plot nodes
    plt.plot(x_interp, y_interp, 'b-', label='Interpolated Function')  # Plot interpolation
    plt.title('Interpolation and Nodes')
    plt.legend()
    
    ''' Now validate the interpolation '''
    Neval = 1000  # Number of evaluation points for the exact function
    xeval = np.linspace(a, b, Neval+1)  # High-resolution points for evaluation
    yeval = np.interp(xeval, x_interp, y_interp)  # Evaluate the interpolated function using numpy's interp
    yex = f(xeval)  # Exact function evaluated at those points
    
    ''' Compute the error '''
    err = yex - yeval  # Error between exact and interpolated function
    errnorm = norm(err)  # Norm of the error
    print('Error norm = ', errnorm)
    
    ''' Plot the exact function and the interpolated function '''
    plt.figure()
    plt.plot(xeval, yeval, label='Interpolated Function', alpha=0.7)
    plt.plot(xeval, yex, label='Exact Function', alpha=0.7)
    plt.title('Exact vs Interpolated Function')
    plt.legend()

    ''' Plot the error on a semilogarithmic plot '''
    plt.figure()
    plt.semilogy(xeval, np.abs(err), label='Error', alpha=0.7)
    plt.title('Error on Semi-log Plot')
    plt.legend()

    plt.show()

def lineval(x0, y0, x1, y1, alpha):
    f = lambda x: y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    return f(alpha)

driver()

    