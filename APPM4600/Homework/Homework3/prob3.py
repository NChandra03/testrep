# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:30:13 2024

@author: Zara Chandra
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt

def driver():

# use routines    
    f = lambda x: x ** 3 + x - 4

    a = 1
    b = 4

    tol = 1e-3

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('attempts', count)
    
    # Plotting
    x = np.linspace(0, 10, 400)  # Generate 400 points between 4 and 6
    y = f(x)  # Compute f(x) for all x
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='$f(x) = (x-5)^9$ expanded form')
    plt.axhline(0, color='black', linewidth=0.8)  # x-axis
    plt.axvline(astar, color='red', linestyle='--', label=f'Approx. root: {astar:.4f}')  # Vertical line at the root
    plt.title('Plot of $f(x)$ and its root in [4, 6]')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()




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
       return [astar, ier, 'pos']

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, 'no']

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier,'no']

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
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
    return [astar, ier, count]
      
driver()  