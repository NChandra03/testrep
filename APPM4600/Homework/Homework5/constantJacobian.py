# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:39:53 2024

@author: Zara Chandra
"""

import numpy as np
import time
from numpy.linalg import inv, norm

def driver():
    x0 = np.array([1, 1])  # Set initial guess to (1,1)
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
        [xstar, ier, its] = Newton(x0, tol, Nmax)
    elapsed = time.time() - t
    print("Solution:", xstar)
    print("Newton: the error message reads:", ier) 
    print("Newton: took this many seconds:", elapsed / 50)
    print("Newton: number of iterations is:", its)

def evalF(x):
    # Define the functions f(x, y) = 3x^2 - y^2 and g(x, y) = 3xy^2 - x^3 - 1
    F = np.zeros(2)
    F[0] = 3 * x[0]**2 - x[1]**2  # f(x, y)
    F[1] = 3 * x[0] * x[1]**2 - x[0]**3 - 1  # g(x, y)
    return F

def evalJ(x):
    # Jacobian of f(x, y) = 3x^2 - y^2 and g(x, y) = 3xy^2 - x^3 - 1
    J = np.array([[6 * x[0], -2 * x[1]], 
                  [3 * x[1]**2 - 3 * x[0]**2, 6 * x[0] * x[1]]])
    return J

def Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approx root, ier = error message, its = num its'''
    
    for its in range(Nmax):
        J = evalJ(x0)  # Evaluate the Jacobian matrix
        Jinv = inv(J)  # Invert the Jacobian matrix
        F = evalF(x0)  # Evaluate the function at the current point
        
        x1 = x0 - Jinv.dot(F)  # Newton-Raphson update
        
        if norm(x1 - x0) < tol:  # Check for convergence
            xstar = x1
            ier = 0
            return [xstar, ier, its]
        
        x0 = x1
    
    xstar = x1
    ier = 1  # Indicate that maximum iterations were reached
    return [xstar, ier, its]

driver()
