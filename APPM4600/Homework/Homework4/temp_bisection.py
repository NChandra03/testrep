# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:19:48 2024

@author: Zara Chandra
"""

import numpy as np
from scipy.special import erf

def driver():

# use routines    

    # Define the parameters
    T_i = 20  # Initial soil temperature [degrees C]
    T_s = -15  # Constant surface temperature [degrees C]
    alpha = 0.138e-6  # Thermal diffusivity [meters^2 per second]
    t = 60 * 24 * 60 * 60  # Time in seconds (60 days)

    # Define the function T(x)
    def f(x):
        return T_s + (T_i - T_s) * erf(x / (2 * np.sqrt(alpha * t)))
    
    a = 0
    b = 1

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-13

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    print("iterations", count+1)
    return [astar, ier]
      
driver()     