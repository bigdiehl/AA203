# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:33:48 2019

@author: diehl
"""

from sympy import *
import numpy as np

init_session()
init_printing(use_latex='mathjax', use_unicode = True)

x,y = symbols('x y')

#%% Calculate Gradients
f1 = Matrix([-log(10-2*x**2-y**2)])
f2 = Matrix([x**2*(1+2*y-x**2)])

var = Matrix([x,y])

grad1 = f1.jacobian(var).T
grad2 = f2.jacobian(var).T


#%% Calculate Hessians

H1 = grad1.jacobian(var)
H2 = grad2.jacobian(var)


#%%Check Eigenvalues at (0,0)
H_1 = np.array([[4.0/10,0],
                [0,2.0/10]])
                
H_2 = np.array([[2.0,0],
                [0,0]])
                

print np.linalg.eig(H_1)
print np.linalg.eig(H_2)

r,D,d = symbols('r D d')

expand((r+D-d)**2)

#%% Problem 
u1,u2,u3 = symbols('u_1 u_2 u_3')
v11, v12, v13 = symbols('v_11 v_12 v_13')
v21, v22, v23, v31, v32, v33 = symbols('v_21 v_22 v_23 v_31 v_32 v_33')


E = Matrix([[u1,0,0],
            [0,u2,0],
            [0,0,u3]])
            
U = Matrix([[v11, v12, v13],
            [v21, v22, v23],
            [v31, v32, v33]])

display(U*E)
display(E*U.T)

display(U*E*E*U.T)



#%% Problem 3

#inequality constraint tight
A = np.array([[1,0,0,1],
              [0,1,0,1],
              [0,0,1,1],
              [1,1,1,0]])
              
y = np.array([0,0,0,-3])

x = np.dot(np.linalg.inv(A),y)
print x

