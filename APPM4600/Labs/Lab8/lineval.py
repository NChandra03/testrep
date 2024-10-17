# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:36:46 2024

@author: Zara Chandra
"""

def lineval(x0,y0,x1,y1,alpha):
    f = lambda x: y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    return (f(alpha))

print(lineval(1,1,2,2,3))