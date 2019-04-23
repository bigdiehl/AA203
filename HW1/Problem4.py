# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:44:13 2019

@author: diehl
"""

import numpy as np
import matplotlib.pyplot as plt

#Parameter
gamma = 10
n = 2

#Gamma = 10 --> 0.2 max constant step size
#Gamma = 2 --> 1.0 max constant step size
def obj(x):
    return 1.0/2*(x[0]**2+gamma*x[1]**2)
    
def grad(x):
    return np.array([x[0],gamma*x[1]])
    
def Hessian(x):
    return np.array([[1.0, 0],
                     [0, gamma]])

def stepSize(x):
    dk = -grad(x)
    Q = Hessian(x)
    return np.linalg.norm(dk)**2/np.linalg.multi_dot((dk,Q,dk))
    
x0_1 = np.array([5.0,1.0])
x0_2 = np.array([1.0,5.0])

def optimize(x0):
    x = x0
    x_record = x0
    i = 0
    while np.linalg.norm(grad(x)) > 1e-3 and i < 1000:
        i+=1
        if(0):
            step = stepSize(x)
        else:
            step = 0.17
            
        x_new = x - step*grad(x)
        x_record = np.vstack((x_record, x_new))
        x = x_new
        
    return x_record
    
    
x_record1 = optimize(x0_1)
x_record2 = optimize(x0_2)

#Plot contours
y = np.arange(-5.0,5.1,0.01)
x = np.arange(-5.0,5.1,0.01)
X,Y = np.meshgrid(x,y)

Z = obj([X,Y])

#Plotting
plt.clf()
plt.contour(X,Y,Z, levels = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125])

plt.plot(x_record1[:,0], x_record1[:,1], 'r')
plt.plot(x_record2[:,0], x_record2[:,1], 'b')
plt.title('Gradient descent with optimal step size')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(('x0 = (5,1)', 'x0 = (1,5)'))



















