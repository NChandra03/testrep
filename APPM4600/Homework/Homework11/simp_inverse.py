# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:56:52 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.close("all")
def trap(f,n):
    #f = lambda x: 1/(1+x ** 2)

    a = -5
    b = 5
    h = (b-a)/n

    total = f(a) + f(b)
    for i in range(1,n):
        total = total + 2 * f(a + h * i)

    total = total * h/2
    return(total)

def simp(f,n):
    #print(n)
    #f = lambda x: 1/(1+x ** 2)

    a = 0
    b = 1
    h = (b-a)/n
    
    #total = f(a) + f(b)
    total = f(b)
    for i in range(1,int(n/2)):
        #print(i * 2,"even")
        total = total + 2 * f(a + h * i * 2)
    
    for i in range(0,int(n/2)):
        #print(i * 2 + 1,"odd")
        total = total + 4 * f(a + h * (i * 2 + 1))
        
    total = total * h/3
    
    return(total)

f = lambda t: np.cos(1/t) * t

print(simp(f,1001))
# =============================================================================
# sol = np.arctan(5) - np.arctan(-5)
# 
# print(abs(trap(409) - sol))
# print(abs(simp(58) - sol))
# 
# f = lambda x: 1/(1+x ** 2)
# result, error = quad(f, -5, 5, epsabs=1e-6)
# print("Result:", result)
# print("Estimated error:", error)
# 
# result, error = quad(f, -5, 5, epsabs=1e-4)
# 
# print("Result:", result)
# print("Estimated error:", error)
# =============================================================================
