# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:10:28 2019

@author: diehl
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import numpy.linalg as LA
from numpy.linalg import multi_dot as dot

np.set_printoptions(precision=2)

dt = 0.1
mc = 10.0
mp = 2.0
l  = 1.0
g  = 9.81

a1 = mp*g/mc
a2 = (mc+mp)*g/(l*mc)

A = np.array([[0,0,1.0,0],
              [0,0,0,1.0],
              [0,a1,0,0],
              [0,a2,0,0]])

B = np.array([[0],[0],[1.0/mc],[1.0/(l*mc)]])

#A and B system is controllable (controllability matrix is full rank)

Q = np.eye(4)
R = np.eye(1)

K,S,E = ctrl.lqr(A,B,Q,R)

print K
print '\n'
print S


#%%
def RecursiveRiccati(A,B,Q,R,N):
    if N == 0:
        return Q
    else:
        P_k = RecursiveRiccati(A,B,Q,R,N-1)
        P_kprev = (dot((A.T,P_k,A))-
            dot((A.T,P_k,B, LA.pinv(R+dot((B.T,P_k,B))), B.T,P_k,A))+Q)
        return P_kprev
        
    

N = 0
#print RecursiveRiccati(A,B,Q,R,N)
#print RecursiveRiccati(A,B,Q,R,N+1)



'''
%Riccati recursion until approximate convergence
    N = 1;
    P_1 = RiccatiRecursion(A,B,Q,R,N);
    L_current = pinv(R + B'*P_1*B)*(B'*P_1*A);
    
    N = 2;
    P_2 = RiccatiRecursion(A,B,Q,R,N);
    L_new = pinv(R + B'*P_2*B)*(B'*P_2*A);
    
    P = P_2;
    
    while norm((L_new-L_current),2) > 1e-4
        L_current = L_new;
        N = N+1;
        P = RiccatiRecursion(A,B,Q,R,N);
        L_new = pinv(R + B'*P*B)*(B'*P*A);
    end
    L = L_new;
    
    
'''

#%%
def RecursiveRiccati(A,B,Q,R,N):
    if N == 0:
        return Q
    else:
        P_k = RecursiveRiccati(A,B,Q,R,N-1)
        L_kprev = -dot((LA.pinv(R+dot((B.T,P_k,B))),dot((B.T,P_k,A))))
        P_kprev = (Q + dot((L_kprev.T,R,L_kprev)) +
                    dot(( (A+np.dot(B,L_kprev)).T, P_k, (A+np.dot(B,L_kprev)) )) )
        return P_kprev
    

N = 0
#print RecursiveRiccati(A,B,Q,R,N)
#print RecursiveRiccati(A,B,Q,R,N+1)
#print RecursiveRiccati(A,B,Q,R,N+500)




#%%Forget this recursive stuff

P_k = Q
for i in range(10):
    P_k = (dot((A.T,P_k,A)) -
                dot((A.T,P_k,B, LA.pinv(R+dot((B.T,P_k,B))), B.T,P_k,A)) + Q)

print '\n'               
print P_k

L_new = dot((LA.pinv(R + dot((B.T,P_k,B))),B.T,P_k,A ))
print '\n',L_new
