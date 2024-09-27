# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:51:59 2024

@author: Zara Chandra
"""

# import libraries
import numpy as np

def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2

  def f(x):
    return np.exp(3*x) - 27*x**6 + 27*x**4 * np.exp(x) - 9*x**2 * np.exp(2*x)
  def fp(x):
    return 3 * np.exp(3*x) - 162 * x**5 + 27 * (4 * x**3 * np.exp(x) + x**4 * np.exp(x)) - 9 * (2 * x * np.exp(2*x) + x**2 * 2 * np.exp(2*x))
  p0 = 4

  Nmax = 100
  tol = 1e-8

  (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  #print(p)
  print(p[4:8])
  compute_order(p[0:4],pstar)


def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
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
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]
        
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

driver()
