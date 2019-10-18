
# coding: utf-8

# In[1]:


#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 26/9-2019
# Lecture: 5  Clustering
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality : 
# ###################################

from scipy.io import loadmat
from numpy import eye, array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

noisy_data = loadmat('l10_y.mat')['y']
clean_data = loadmat('l10_x_true.mat')['x_true']

noisy_data.shape

EKF = ExtendedKalmanFilter(1,1)
EKF.x = 0

dt = 0.05
EKF.F = eye(3) + array([[0, 1, 0],[0, 0, 0], [0, 0, 0]])*dt
EKF.R

def HJacobian_at(x):
    new_signal = (1.9999*x[i-1])+(-1*x[i-2])
    return new_signal


#EKF.y = noisy_data
#print(EKF.x_post)
#something = EKF.update(None,'HJacobian',clean_data)
plt.scatter(range(0,6234),noisy_data,c='b')
plt.scatter(range(0,6234),clean_data,c='r')
#plt.show()