#!.venv/bin/python3
#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham,
# Date : 11/9-2019
# Lecture: 3  Parametric and nonparametric methods

from scipy.io import loadmat
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
mat_file = loadmat("dataset1_G_noisy.mat")

classes = (mat_file["trn_x"], mat_file["trn_y"])

train_x_class = mat_file["trn_x_class"]
train_y_class = mat_file["trn_y_class"]

"""
data_normal = norm.rvs(size=1000,loc=0,scale=1)
ax = sns.distplot(data_normal,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
"""

def task_a():
    """
    Classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy
    """
    
    test_xy = mat_file["tst_xy"]
    pprint(mat_file["trn_x"][0:])
    plt.scatter(mat_file["trn_x"][:,0],mat_file["trn_x"][:,1])
    plt.scatter(mat_file["trn_y"][:,0],mat_file["trn_y"][:,1])
    plt.show()

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