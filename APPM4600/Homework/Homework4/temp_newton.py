# import libraries
import numpy as np
from scipy.special import erf
        
def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2

  # Define the parameters
  T_i = 20  # Initial soil temperature [degrees C]
  T_s = -15  # Constant surface temperature [degrees C]
  alpha = 0.138e-6  # Thermal diffusivity [meters^2 per second]
  t = 60 * 24 * 60 * 60  # Time in seconds (60 days)

  # Define the function T(x)
  def f(x):
      return T_s + (T_i - T_s) * erf(x / (2 * np.sqrt(alpha * t)))
  
  def fp(x):
    coefficient = (T_i - T_s) / np.sqrt(np.pi * alpha * t)
    exponential_term = np.exp(- (x / (2 * np.sqrt(alpha * t))) ** 2)
    return coefficient * exponential_term
  
  p0 = 1

  Nmax = 100
  tol = 1e-13

  (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  print('f(pstar) =', f(pstar))


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
        
driver()
