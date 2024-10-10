import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import norm

def driver():

    plt.close('all')
    f = lambda x: 1/(1 + (10 * x) ** 2)
    
    N = 12
    a = -1
    b = 1
    
    h = 2 / (N - 1)
    ''' Create interpolation nodes'''
    #xint = np.array([-1 + (j - 1) * h for j in range(N + 1)])
    xint = np.array([np.cos((2 * j - 1) * np.pi / (2 * N)) for j in range(N + 1)])
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex, label='function')
    plt.plot(xeval,yeval_l,alpha = 0.5, label='lagrange') 
    plt.plot(xeval,yeval_dd,alpha = 0.5,label='Newton DD')
    plt.title('interp')
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    print(norm(err_l))
    err_dd = abs(yeval_dd-fex)
    print(norm(err_dd))
    plt.semilogy(xeval,err_l,label='lagrange',alpha = 0.5)
    plt.semilogy(xeval,err_dd,label='Newton DD',alpha = 0.5)
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
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       

driver()        
