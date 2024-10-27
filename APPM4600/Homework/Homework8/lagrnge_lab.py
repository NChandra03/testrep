import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import norm

def driver(N):

    plt.close('all')
        
    #N = 20
    
    f = lambda x: 1/(1+x**2) # Exact function
    a = -5  # Interval start
    b = 5   # Interval end
    
    h = (b-a) / N
    #xint = np.array([a + j * h for j in range(N + 1)])  # N+1 points, 0 to N inclusive
    #xint = np.linspace(a,b,N+1)
    
    xint = np.array([np.cos((2 * j - 1) * np.pi / (2 * (N + 1))) for j in range(1, N + 2)])  # N+1 points in [-1, 1]
    xint = 0.5 * (a + b) + 0.5 * (b - a) * xint  # Map from [-1, 1] to [a, b]
    xint = xint[::-1]

    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex, label='function')
    plt.plot(xeval,yeval_l,alpha = 0.5, label='lagrange') 
    plt.title('interp')
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    print(N,':', norm(err_l))
    plt.semilogy(xeval,err_l,label='lagrange',alpha = 0.5)
    plt.title('error')
    plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  

       

driver(5) 
driver(10)
driver(15)
driver(20)       
