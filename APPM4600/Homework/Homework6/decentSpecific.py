# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:38:34 2024

@author: Zara Chandra
"""

#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():

    Nmax = 100
    x0 = np.array([0.1,0.1,0.1])
    tol = 1e-6
    
    [xstar,gval,ier,its] = SteepestDescent(x0,tol,Nmax)
    print("the steepest descent code found the solution ",xstar)
    print("g evaluated at this point is ", gval)
    print("ier is ", ier	)
    print("its is ", its	)

###########################################################
#functions:
def evalF(x): 
    F = np.zeros(3)
    
    # Equation 1: x + cos(xyz) - 1
    F[0] = x[0] + math.cos(x[0] * x[1] * x[2]) - 1

    # Equation 2: (1 - x)^(1/4) + y + 0.05*z^2 - 0.15*z - 1
    F[1] = (1 - x[0])**(1/4) + x[1] + 0.05 * x[2]**2 - 0.15 * x[2] - 1

    # Equation 3: -x^2 - 0.1*y^2 + 0.01*y + z - 1
    F[2] = -x[0]**2 - 0.1 * x[1]**2 + 0.01 * x[1] + x[2] - 1
    
    return F

    
def evalJ(x): 
    J = np.zeros((3, 3))
    
    # Partial derivatives for F1: x + cos(xyz) - 1
    J[0, 0] = 1 - x[1] * x[2] * math.sin(x[0] * x[1] * x[2])  # dF1/dx
    J[0, 1] = -x[0] * x[2] * math.sin(x[0] * x[1] * x[2])     # dF1/dy
    J[0, 2] = -x[0] * x[1] * math.sin(x[0] * x[1] * x[2])     # dF1/dz

    # Partial derivatives for F2: (1 - x)^(1/4) + y + 0.05*z^2 - 0.15*z - 1
    J[1, 0] = -0.25 * (1 - x[0])**(-3/4)                      # dF2/dx
    J[1, 1] = 1                                                # dF2/dy
    J[1, 2] = 0.1 * x[2] - 0.15                                # dF2/dz

    # Partial derivatives for F3: -x^2 - 0.1*y^2 + 0.01*y + z - 1
    J[2, 0] = -2 * x[0]                                        # dF3/dx
    J[2, 1] = -0.2 * x[1] + 0.01                               # dF3/dy
    J[2, 2] = 1                                                # dF3/dz

    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]



if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
