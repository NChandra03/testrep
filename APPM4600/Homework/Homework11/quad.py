# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:43:03 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.close("all")
def trap(n):
    f = lambda x: 1/(1+x ** 2)

    a = -5
    b = 5
    h = (b-a)/n

    total = f(a) + f(b)
    for i in range(1,n):
        total = total + 2 * f(a + h * i)

    total = total * h/2
    return(total)

def simp(n):
    #print(n)
    f = lambda x: 1/(1+x ** 2)

    a = -5
    b = 5
    h = (b-a)/n
    
    total = f(a) + f(b)
    for i in range(1,int(n/2)):
        #print(i * 2,"even")
        total = total + 2 * f(a + h * i * 2)
    
    for i in range(0,int(n/2)):
        #print(i * 2 + 1,"odd")
        total = total + 4 * f(a + h * (i * 2 + 1))
        
    total = total * h/3
    
    return(total)

sol = np.arctan(5) - np.arctan(-5)

print(abs(trap(409) - sol))
print(abs(simp(58) - sol))

f = lambda x: 1/(1+x ** 2)
result, error = quad(f, -5, 5, epsabs=1e-6)
print("Result:", result)
print("Estimated error:", error)

result, error = quad(f, -5, 5, epsabs=1e-4)

print("Result:", result)
print("Estimated error:", error)

# =============================================================================
# x1 = []
# x2 = []
# num = []
# for n in range(2,52,2):
#     num.append(n)
#     x1.append(abs(trap(n) - sol))
#     x2.append(abs(simp(n) - sol))
#     
# plt.plot(num,x1,label = 'trapazoidal')
# plt.plot(num,x2,label = 'simpsons')
# plt.legend()
# plt.yscale('log')
# plt.title('error vs intervals')
# =============================================================================
