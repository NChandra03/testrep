# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:08:37 2024

@author: Zara Chandra
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
        
def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2

      # Define the function f(x)
  def f(x):
        return x**6 - x - 1
    
  p0 = 2
  p1 = 1  

  Nmax = 100
  tol = 1e-14

  (p,pstar,info,it) = secant(f,p0,p1,tol, Nmax)
  alpha = 1.1347241384015194  # Exact root approximation
  
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  print(p[0:8])
  
  # Plot log-log of (p - alpha) vs (p - alpha) shifted by 1
  plot_log_log_p_minus_alpha(p, alpha, it)
    
def secant(f, p0, p1, tol, Nmax):
    """
    Secant iteration.
    
    Inputs:
        f    - function
        p0   - first initial guess for root
        p1   - second initial guess for root
        tol  - iteration stops when p_n and p_{n+1} are within tol
        Nmax - max number of iterations
    Returns:
        p     - an array of the iterates
        pstar - the last iterate
        info  - success message
              - 0 if we met tol
              - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax + 1)
    p[0] = p0
    p[1] = p1
    for it in range(1, Nmax):
        # Calculate the next approximation using the Secant method formula
        pn = p[it] - f(p[it]) * (p[it] - p[it - 1]) / (f(p[it]) - f(p[it - 1]))
        p[it + 1] = pn
        
        # Check for convergence
        if abs(pn - p[it]) < tol:
            pstar = pn
            info = 0
            return [p, pstar, info, it]
    
    # If maximum iterations are reached without convergence
    pstar = pn
    info = 1
    return [p, pstar, info, it]

def plot_log_log_p_minus_alpha(p, alpha, it):
    """
    Plot log-log of (p - alpha) vs. (p - alpha) shifted by 1.

    Inputs:
      p     - array of iterates
      alpha - exact root approximation
      it    - number of iterations
    """
    # Calculate (p - alpha) for each iteration
    p_minus_alpha = abs(p[:it+1] - alpha)
    x = p_minus_alpha[:-1]  # p - alpha for n
    y = p_minus_alpha[1:]   # p - alpha for n+1
    print(p_minus_alpha)

    # Filter out zero values to avoid log(0) issues
    non_zero_indices = (x != 0) & (y != 0)
    x_non_zero = x[non_zero_indices]
    y_non_zero = y[non_zero_indices]

    print(x_non_zero)
    print(y_non_zero)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(x_non_zero, y_non_zero, 'o-', label='$p_{n+1} - \\alpha$ vs. $p_n - \\alpha$')
    plt.xlabel('$p_n - \\alpha$')
    plt.ylabel('$p_{n+1} - \\alpha$')
    plt.title('Log-Log Plot of $(p_n - \\alpha)$ vs. $(p_{n+1} - \\alpha)$ for Secant')
    plt.axis('equal')  # Ensure equal axis scaling
    plt.legend()
    plt.xlim(1e-6, 1)  # Set x-axis limits for better visualization
    plt.ylim(1e-10, 1)  # Set y-axis limits for better visualization
    plt.grid(True, which="both", ls="--")
    plt.show()
        
driver()