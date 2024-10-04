# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:34:04 2024

@author: Zara Chandra
"""

import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([-1,0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
    
    t = time.time()
    for j in range(20):
        [xstar, ier, its] = SlackerNewton(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Slacker Newton: the error message reads:', ier)
    print('Slacker Newton: took this many seconds:', elapsed / 20)
    print('Slacker Newton: number of iterations is:', its)
    
def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]


def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]  

def SlackerNewton(x0, tol, Nmax):
    ''' Slacker Newton = re-evaluate the Jacobian every 5 iterations '''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its '''
    ''' Outputs: xstar = approx root, ier = error message, its = num its '''
    
    J = evalJ(x0)  # Initial Jacobian
    Jinv = inv(J)   # Inverse of the Jacobian
    for its in range(Nmax):
        
        if its % 5 == 0:  # Recompute the Jacobian every 5 iterations
            J = evalJ(x0)
            Jinv = inv(J)
        
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        
        if norm(x1 - x0) < tol:  # Check for convergence
            xstar = x1
            ier = 0
            return [xstar, ier, its]
        
        x0 = x1  # Update x0 for the next iteration
    
    xstar = x1  # Return final estimate if max iterations reached
    ier = 1
    return [xstar, ier, its]


def evalF(x):
    F = np.zeros(2)
    
    # First equation: 4*x1^2 + x2^2 - 4
    F[0] = 4 * x[0]**2 + x[1]**2 - 4
    
    # Second equation: x1 + x2 - sin(x1 - x2)
    F[1] = x[0] + x[1] - math.sin(x[0] - x[1])
    
    return F
    
def evalJ(x):
    J = np.zeros((2, 2))
    
    # First row: partial derivatives of the first equation
    J[0, 0] = 8 * x[0]
    J[0, 1] = 2 * x[1]
    
    # Second row: partial derivatives of the second equation
    J[1, 0] = 1 - math.cos(x[0] - x[1])
    J[1, 1] = 1 + math.cos(x[0] - x[1])
    
    return J

driver()