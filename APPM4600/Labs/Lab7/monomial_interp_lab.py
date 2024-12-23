import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def driver(): 
    plt.close('all')

    f = lambda x: 1/(1 + (10 * x) ** 2)
    
    N = 2
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    h = 2 / (N - 1)
    xint = np.array([-1 + (j - 1) * h for j in range(N + 1)])
    print('xint =',xint)

    xint = np.array([np.cos((2 * j - 1) * np.pi / (2 * N)) for j in range(N + 1)])
 
    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    #print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
#    print('coef = ', coef)

# No validate the code
    Neval = 1000 
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  yex-yeval
    errnorm =  norm(yex-yeval) 
    print('err = ', errnorm)
    
    plt.figure() 
    plt.plot(xeval,yeval)
    plt.plot(xeval,yex)
    plt.title('functions')
    
    plt.figure() 
    plt.semilogy(xeval,err,label='error',alpha = 0.5)
    plt.title('error')
    plt.legend()
    plt.show()
    
    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

driver()    
