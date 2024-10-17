# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:55:51 2024

@author: Zara Chandra
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    plt.close('all')
    f = lambda x: 1/(1 + (10 * x) ** 2)  # Exact function
    a = -1  # Interval start
    b = 1   # Interval end
    
    ''' number of intervals'''
    Nint = 12
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

#   Create the coefficients for the natural spline    
    (M,C,D) = create_natural_spline(yint,xint,Nint)

#  evaluate the cubic spline     
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,label='exact function')
    plt.plot(xeval,yeval,label='natural spline') 
    plt.legend
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,label='absolute error')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint, xint, N):

    # Initialize h and b vectors
    h = np.zeros(N)
    b = np.zeros(N+1)  # b should have size N+1 as you want to solve for M
    
    # Calculate h values (spacing between consecutive x points)
    for i in range(N):
        h[i] = xint[i+1] - xint[i]  # h[i] = x_{i+1} - x_i
    
    # Calculate b values (spline slope-related terms)
    for i in range(1, N):
        b[i] = (yint[i+1] - yint[i]) / h[i] - (yint[i] - yint[i-1]) / h[i-1]

    # Set natural boundary conditions (b[0] and b[N] should be 0)
    b[0] = 0
    b[-1] = 0

    # Create the matrix A to solve for the M values
    A = np.zeros((N+1, N+1))  # A is the tridiagonal matrix with size (N+1) x (N+1)

    # Fill the main diagonal and off-diagonals of A
    for i in range(1, N):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    # Set the natural boundary conditions in the matrix A
    A[0, 0] = 1  # First row for natural boundary at the start
    A[N, N] = 1  # Last row for natural boundary at the end

    # Solve for M (second derivatives) using numpy's built-in solver
    M = np.linalg.solve(A, b)

    # Initialize C and D coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    
    # Compute C and D coefficients
    for j in range(N):
        C[j] = (yint[j] / h[j]) - (h[j] * M[j] / 6)  # C_j formula
        D[j] = (yint[j+1] / h[j]) - (h[j] * M[j+1] / 6)  # D_j formula
        
    return M, C, D
       
def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluates the local spline as defined in class
    # xip = x_{i+1}; xi = x_i
    # Mip = M_{i+1}; Mi = M_i
    
    hi = xip - xi  # Compute h_i

    # Implement the spline evaluation formula
    yeval = ((xip - xeval)**3 * Mi) / (6 * hi) + ((xeval - xi)**3 * Mip) / (6 * hi) \
            + C * (xip - xeval) + D * (xeval - xi)

    return yeval
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()               