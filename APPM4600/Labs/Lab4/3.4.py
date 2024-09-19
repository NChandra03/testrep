# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:57:08 2024

@author: Zara Chandra
"""

import numpy as np

def driver():

# test functions 
     # Define the function g(x)
     def g(x):
         return np.sqrt(10 / (x + 4))

     # Calculate the value of p
     p = 1.3652300134140976

     Nmax = 100
     tol = 1e-10
     
     x0 = 1.5
     
     [xstar,ier,attempts] = Steffenson(g,x0,tol,Nmax)
     
     #print(len(attempts))
     
     print(attempts)
     
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',g(xstar))
     print('Error message reads:',ier)
     
     compute_order(attempts,p)
     
     
def compute_order(x, xstar):
    """
    Approximates order of convergence given:
    x: array of iterate values
    xstar: fixed point/solution
    """
    # |x_n+1 - x*|
    diff1 = np.abs(x[1::]-xstar)
    # |x_n - x*|
    diff2 = np.abs(x[0:-1]-xstar)
    
    # take linear fit of logs to find slope (alpha) and intercept (lambda)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    
    print(f"lambda is {_lambda}")
    print(f"alpha is {alpha}")
    
    return fit

def Steffenson(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    attempts = []
    count = 0
    while (count <Nmax):
       
       a = x0
       b = f(x0)
       c = f(b)
       x1 = a - ((b - a) ** 2 / (c - 2 * b + a))
       attempts.append(x1)
       
       count = count +1
       
       if (abs(x1-x0) < tol):
          xstar = x1
          ier = 0
          return [xstar,ier,np.array(attempts)]
      
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, np.array(attempts)]

  
driver()