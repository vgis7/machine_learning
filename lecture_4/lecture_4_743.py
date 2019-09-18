#!.venv/bin/python3
#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, 
# Date : 18/9-2019
# Lecture: 4  Dimensionality reduction
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality : This script trains two likelyhood estimators on
#                 two classes of points, and then tests these on annotated test points
# ###################################

#5,6,8 <-- 
# Do exercise2:  (1) from the 10-class database, choose three classes (5, 6 and 8) 
# and then reduce dimension to 2; (2) perform 3-class classification based on the 
# generated 2-dimensional data

# Mat file (train0-9 and test 0-9)

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.decomposition import PCA
import cv2


mat_file = loadmat("mnist_all.mat")

train = []
test = []
reduced = []

for idx in range(0,10):
    train.append(mat_file[f"train{idx}"])
    test.append(mat_file[f"test{idx}"])


def task_1():
    global train
    global test
    global reduced
    pcas = []
    for item in train:
        pcas.append(PCA(n_components=2))
        pcas[-1].fit(item)
        reduced.append(pcas[-1].transform(item))
    

    for idx, item in enumerate(pcas):
        mean = item.mean_
        plt.figure(idx+1)
        resized_image = np.resize(mean,(28,28))
        plt.imshow(resized_image)
    #plt.show()

def task_2():
    pass

task_1()