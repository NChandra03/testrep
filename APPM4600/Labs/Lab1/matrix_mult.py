# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:06:17 2024

@author: Zara Chandra
"""
import numpy as np

def driver():
    #This is the main section of the code
    #matrix1 = np.array([[1, 2], [3, 4]])
    #matrix2 = np.array([[5, 6], [7, 8]])
    # Defining Matrix 1
    matrix1 = np.array([[1, 4, -2],
                    [3, 5, -6]])

    # Defining Matrix 2
    matrix2 = np.array([[5, 2, 8, -1],
                    [3, 6, 4, 5],
                    [-2, 9, 7, -3]])
    ans = matrixMult(matrix1, matrix2)
    #This is the numpy method for matrix mult for checking
    #ans = matrix1 @ matrix2
    return ans

def matrixMult(matrix1, matrix2):
    d1 = len(matrix1)
    d2 = len(matrix2[0])
    n = len(matrix1[0])
    ans = np.zeros((d1,d2))
    for i in range (d1):
        for j in range (d2):
            ans[i,j] = dotProduct(matrix1[i, :], matrix2[: , j], n)
    return ans

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

ans = driver()