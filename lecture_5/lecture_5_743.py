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
import pandas as pd

from sklearn.cluster import KMeans


mat_file = loadmat("lecture_5/2D3classes.mat") #Loading the matfile with the 2d data for the 3 classes

X = np.concatenate((mat_file['trn5'],mat_file['trn6'],mat_file['trn8'])) #Adds all three digits to one numpy array for the train dataset

# Loads the data into 3 variables
trainSV5 =  mat_file['trn5']
trainSV6 =  mat_file['trn6']
trainSV8 =  mat_file['trn8']

#Calculates the variance for the labeled data
varTrainSV5 = np.var(trainSV5)
varTrainSV6 = np.var(trainSV6)
varTrainSV8 = np.var(trainSV8)

#Calculates the mean for the labeled data
meanTrainSV5 = (np.mean(trainSV5[:,0]),np.mean(trainSV5[:,1]))
meanTrainSV6 = (np.mean(trainSV6[:,0]),np.mean(trainSV6[:,1]))
meanTrainSV8 = (np.mean(trainSV8[:,0]),np.mean(trainSV8[:,1]))


print("varinace Class5: "+str(varTrainSV5))
print("varinace Class6: "+str(varTrainSV6))
print("varinace Class8: "+str(varTrainSV8))


#Clusters the unlabeled data by Kmeans
kmean = KMeans(n_clusters=3)
kmean.fit(X)

#Sorts the culsters into new lists
TempCluster1=[]
TempCluster2=[]
TempCluster3=[]
for id,x in enumerate(X):
    if kmean.labels_[id]==0:
        TempCluster1.append(x)
    if kmean.labels_[id]==1:
        TempCluster2.append(x)
    if kmean.labels_[id]==2:
        TempCluster3.append(x)

# Makes the lists into arrays
Cluster1 = np.array(TempCluster1)
Cluster2 = np.array(TempCluster2)
Cluster3 = np.array(TempCluster3)

#Calculates variance of the 3 clusters
varCluster1 = np.var(Cluster1)
varCluster2 = np.var(Cluster2)
varCluster3 = np.var(Cluster3)

#Print variance 
print("Variance Cluster1: "+str(varCluster1))
print("Variance Cluster2: "+str(varCluster2))
print("Variance Cluster3: "+str(varCluster3))

#Plots all the datapoints, and the means for the labeled data and the clustered
plt.scatter(X[:,0],X[:,1])
plt.scatter(kmean.cluster_centers_[0,0],kmean.cluster_centers_[0,1],c="y",marker="$c1$")
plt.scatter(kmean.cluster_centers_[1,0],kmean.cluster_centers_[1,1],c="y",marker="$c2$")
plt.scatter(kmean.cluster_centers_[2,0],kmean.cluster_centers_[2,1],c="y",marker="$c3$")
plt.scatter(meanTrainSV5[0],meanTrainSV5[1],c="r",marker="$5$")
plt.scatter(meanTrainSV6[0],meanTrainSV6[1],c="r",marker="$6$")
plt.scatter(meanTrainSV8[0],meanTrainSV8[1],c="r",marker="$8$")
plt.show()
