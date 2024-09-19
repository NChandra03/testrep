# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 22:16:04 2024

@author: Zara Chandra
"""
import numpy as np
def driver():

# test functions 
     a = lambda x: x * (1 + (7 - x**5) / x**2) ** 3

     b = lambda x: x - (x**5 - 7) / x**2
    
     c = lambda x: x - (x**5 - 7) / (5 * x**4)
    
     d = lambda x: x - (x**5 - 7) / 12


     Nmax = 10
     tol = 1e-10

     x0 = 1
     [xstar,ier,attempts] = fixedpt(c,x0,tol,Nmax)
     print(attempts)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',d(xstar))
     print('Error message reads:',ier)
     
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    attempts = np.zeros(Nmax)
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       attempts[count - 1] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier,attempts]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, attempts]

driver()