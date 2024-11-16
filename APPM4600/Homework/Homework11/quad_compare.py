# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:43:41 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.close("all")

# Wrapper to count function calls
def function_counter(f):
    def wrapper(x):
        wrapper.calls += 1
        return f(x)
    wrapper.calls = 0
    return wrapper

def trap(f, n):
    a = -5
    b = 5
    h = (b - a) / n

    total = f(a) + f(b)
    for i in range(1, n):
        total += 2 * f(a + h * i)

    total *= h / 2
    return total

def simp(f, n):
    a = -5
    b = 5
    h = (b - a) / n

    total = f(a) + f(b)
    for i in range(1, int(n / 2)):
        total += 2 * f(a + h * i * 2)
    for i in range(0, int(n / 2)):
        total += 4 * f(a + h * (i * 2 + 1))

    total *= h / 3
    return total

sol = np.arctan(5) - np.arctan(-5)
f = lambda x: 1 / (1 + x ** 2)

# Wrap the function to track calls for trapezoidal and Simpson's methods
f_trap = function_counter(f)
print("Trapezoidal method:")
trap_result = trap(f_trap, 409)
print("Result:", trap_result)
print("Error:", abs(trap_result - sol))
print("Function calls:", f_trap.calls)

f_simp = function_counter(f)
print("\nSimpson's method:")
simp_result = simp(f_simp, 58)
print("Result:", simp_result)
print("Error:", abs(simp_result - sol))
print("Function calls:", f_simp.calls)

# Wrap the function for quad integration and track calls
f_quad = function_counter(f)
print("\nquad method with epsabs=1e-6:")
result, error = quad(f_quad, -5, 5, epsabs=1e-6)
print("Result:", result)
print("Estimated error:", error)
print("Function calls:", f_quad.calls)

# Reset the counter for a different accuracy
f_quad.calls = 0
print("\nquad method with epsabs=1e-4:")
result, error = quad(f_quad, -5, 5, epsabs=1e-4)
print("Result:", result)
print("Estimated error:", error)
print("Function calls:", f_quad.calls)
