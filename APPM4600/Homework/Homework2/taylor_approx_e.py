# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:51:43 2024

@author: Zara Chandra
"""

import math
import numpy as np
import matplotlib.pyplot as plt

x = 9.999999995e-10


def f(x):
    y = math.exp(x)
    print(y)
    return (y - 1)


def t(x):
    y = x + x ** 2/2
    return (y)

def a(x):
    y = math.expm1(x)
    print(y)
    return (y - 1)

ans = f(x) * 10e16
ans2 = t(x) * 10e16
ans3 = a(x) * 10e16
 
# x = np.linspace(-1, 1, 1000)
# 
# plt.plot(x, f(x))
# =============================================================================
