# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:04:16 2019

@author: diehl
"""

from __future__ import division, print_function
 
import numpy as np
import scipy.linalg
      
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
 
    return K, X, eigVals



K,X,eigvals = dlqr(A,B,Q,R)

print(K)
print('\n')
print(X)