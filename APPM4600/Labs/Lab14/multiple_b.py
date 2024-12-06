# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:56:43 2024

@author: Zara Chandra
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy.linalg import lu_factor, lu_solve
import time

def driver(N,n):

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

# =============================================================================
#     start_time = time.time()
#     time.sleep(1)
#     end_time = time.time()
#     print("Execution time:", end_time - start_time, "seconds")
# =============================================================================
 
     ''' Right hand side'''
     start_time_scila = time.time()
     for i in range (n):
         b = np.random.rand(N,1)
         A = np.random.rand(N,N)
         x = scila.solve(A,b)
     end_time_scila = time.time()
     
     print("Execution time scila :", end_time_scila - start_time_scila, "seconds", N)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     #print(r) 
     
     start_time_lufac = time.time()
     decomp = lu_factor(A)
     end_time_lufac = time.time()
     
     start_time_lusolve = time.time()
     for i in range (n):
         x = lu_solve(decomp,b)
     end_time_lusolve = time.time()
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print("Execution time lu :", end_time_lusolve - start_time_lufac, "seconds", N)
     
     print("Execution time lufac :", end_time_lufac - start_time_lufac, "seconds", N)
     print("Execution time lusolve :", end_time_lusolve - start_time_lusolve, "seconds", N)
     print("------------------")
     
     #print(r) 
     
     
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      #driver()       
      trials = [100,500,1000,2000,4000,5000]
      for N in trials:
          driver(N,1000)