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
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


mat_file = loadmat("2D3classes.mat") #Loading the matfile with the 2d data for the 3 classes

X = np.concatenate((mat_file['trn5'],mat_file['trn6'],mat_file['trn8'])) #Adds all three digits to one numpy array for the train dataset

trainSV5 =  mat_file['trn5']
trainSV6 =  mat_file['trn6']
trainSV8 =  mat_file['trn8']

stdTrainSV5 = np.std(trainSV5)
stdTrainSV6 = np.std(trainSV6)
stdTrainSV8 = np.std(trainSV8)

meanTrainSV5 = (np.mean(trainSV5[:,0]),np.mean(trainSV5[:,1]))
meanTrainSV6 = (np.mean(trainSV6[:,0]),np.mean(trainSV6[:,1]))
meanTrainSV8 = (np.mean(trainSV8[:,0]),np.mean(trainSV8[:,1]))


print(stdTrainSV5)
print(meanTrainSV8)


kmean = KMeans(n_clusters=3)
kmean.fit(X)

plt.scatter(X[:,0],X[:,1])
plt.scatter(kmean.cluster_centers_[0,0],kmean.cluster_centers_[0,1],c="y",marker="$c1$")
plt.scatter(kmean.cluster_centers_[1,0],kmean.cluster_centers_[1,1],c="y",marker="$c2$")
plt.scatter(kmean.cluster_centers_[2,0],kmean.cluster_centers_[2,1],c="y",marker="$c3$")
plt.scatter(meanTrainSV5[0],meanTrainSV5[1],c="r",marker="$5$")
plt.scatter(meanTrainSV6[0],meanTrainSV6[1],c="r",marker="$6$")
plt.scatter(meanTrainSV8[0],meanTrainSV8[1],c="r",marker="$8$")
plt.show()
