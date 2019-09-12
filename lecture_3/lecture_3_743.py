#!.venv/bin/python3
#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham,
# Date : 11/9-2019
# Lecture: 3  Parametric and nonparametric methods

from scipy.io import loadmat
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
mat_file = loadmat("dataset1_G_noisy.mat")

classes = (mat_file["trn_x"], mat_file["trn_y"])

train_x_class = mat_file["trn_x_class"]
train_y_class = mat_file["trn_y_class"]

fig, ax = plt.subplots(1, 1)

def task_a():
    """
    Classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy
    """
    
    test_xy = mat_file["tst_xy"]
    pprint(mat_file["trn_x"][0:])
    #plt.scatter(mat_file["trn_x"][:,0],mat_file["trn_x"][:,1])
    #plt.scatter(mat_file["trn_y"][:,0],mat_file["trn_y"][:,1])
    
    #Standard div, mean and variance of x:
    stdx = np.std(mat_file["trn_x"])
    meanx = np.mean(mat_file["trn_x"])
    variancex = np.var(mat_file["trn_x"])

    #Standard div, mean and variance of y:
    stdy = np.std(mat_file["trn_y"])
    meany = np.mean(mat_file["trn_y"])
    variancey = np.var(mat_file["trn_y"])

    print(stdx)
    print(meanx)
    print(variancex)
    print(stdy)
    print(meany)
    print(variancey)
    
    hest = np.array(mat_file["trn_x"]).T

    cov = np.cov(hest)
    print(cov)
    print(mat_file["trn_x"].shape)

    print("Hest is:\t{}\ntrainx is:\t{}".format(hest.shape,mat_file["trn_x"].shape))

    
    #multi = mvn.pdf(test_xy,0,cov)
    #print(multi)
    #plt.show()
    
    #sns.scatterplot(x="x",y="y",data=sns.load_dataset(mat_file["trn_x"]))
    

def task_b():
    """
    Classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy
    """

def task_c():
    """
    classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).
    """

task_a()