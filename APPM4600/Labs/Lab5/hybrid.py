# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:42:38 2024

@author: Zara Chandra
"""

# import libraries
import numpy as np
import math

def driver():

# use routines    
    f = lambda x: math.exp(x**2 + 7*x - 30) - 1
    df = lambda x: math.exp(x**2 + 7*x - 30) * (2*x + 7)
    ddf = lambda x: math.exp(x**2 + 7*x - 30) * ((2*x + 7)**2 + 2)

    a = 2
    b = 4.5

    tol = 1e-8

    [astar,ier] = hybrid(f,df,ddf,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def hybrid(f,df,ddf,a,b,tol):
    
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
    print(f(d)*ddf(d)/df(d) ** 2)
    while (f(d)*ddf(d)/df(d) ** 2 >= 1):
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
    
    Nmax = 100
    p,pstar,info,it = newton(f,df,d,tol,Nmax)
    ier = info
    print(it)
    print(count + 1)
    return [pstar, ier]

def newton(f,df,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,df - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/df(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]
        
driver()
