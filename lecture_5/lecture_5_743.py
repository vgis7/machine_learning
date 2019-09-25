#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 18/9-2019
# Lecture: 5  Clustering
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality :
# ###################################

# First, mix the 2-dimensional data (training data only) by removing the labels and then use one 
# Gaussian mixture model to model them. Secondly, compare the Gaussian mixture model with the Gaussian 
# models trained in the previous assignment, in terms of mean and variance values as well as through visualisation.
from scipy.io import loadmat
import numpy as np



mat_file = loadmat("2D3classes.mat") #Loading the matfile with the 2d data for the 3 classes

train = np.concatenate((mat_file['trn5'],mat_file['trn6'],mat_file['trn8'])) #Adds all three digits to one numpy array for the train dataset

print(train.shape)