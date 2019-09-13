#!.venv/bin/python3
#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham,
# Date : 11/9-2019
# Lecture: 3  Parametric and nonparametric methods

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.io

#Loads the mat file
mat_file = scipy.io.loadmat('dataset1_G_noisy.mat')

train_x = mat_file["trn_x"]
train_y = mat_file["trn_y"]

test_xy = mat_file["tst_xy"]
test_xy_class = mat_file["tst_xy_class"]

#Priors for class 1 and class 2
prior_C1 = len(train_x)/(len(train_x)+len(train_y))
prior_C2 = len(train_y)/(len(train_y)+len(train_x))
print("Prior values for class1: {} and class2: {}".format(round(prior_C1,4),round(prior_C2,4)))

#Mean Values
mean_C1 = (np.mean(train_x[:,0]),np.mean(train_x[:,1]))
mean_C2 = (np.mean(train_y[:,0]),np.mean(train_y[:,1]))

#Standard Diviation
std_C1 = (np.std(train_x[:,0]),np.std(train_x[:,1]))
std_C2 = (np.std(train_y[:,0]),np.std(train_y[:,1]))

#Variance
var_C1 = np.var(train_y) #Not in use
var_C2 = np.var(train_y) #Not in use

#Covariance Matrix
covM_C1 = np.cov(train_x[:,0],train_x[:,1])
covM_C2 = np.cov(train_y[:,0],train_y[:,1])

#Function to classify new data
def classify(x):
  #Posterior = likelihood*prior. MATH: P(x|c1)*P(c1)
  p_C1 = (norm.pdf(x[0],mean_C1[0],std_C1[0])*
          norm.pdf(x[1],mean_C1[1],std_C1[1])*
          prior_C1)

  p_C2 = (norm.pdf(x[0],mean_C2[0],std_C2[0])*
          norm.pdf(x[1],mean_C2[1],std_C2[1])*
          prior_C2)

  #Returns the class value
  if p_C1 > p_C2:
    return 1
  else:
    return 2



def task_a():
    """
    Classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy
    """

    predictions = pd.DataFrame(index=np.arange(0,len(test_xy)),columns=["Class"]) #Initialize empty dataframe that consists of the length of the test data
    for index,data_row in enumerate(test_xy): #Goes through each row in test_xy
        predictions.at[index,"Class"] = classify(data_row) #Classifies each row in test_xy and sets the returned class value to the dataframe of predictions

    correct_counter = 0 #Counts the number of times the classification is correct
    for index,data_row in enumerate(test_xy): #Goes through each row in test_xy
      if predictions.at[index,"Class"] == test_xy_class[index][0]: #Checks if the predicted classification is the same as the true classification
        correct_counter += 1

    accuracy = correct_counter/len(test_xy) #Calculates the accuracy of how many correct classifications that have been done
    print("Correct Predictions: {}. False Predictions: {}.".format(correct_counter,len(test_xy)-correct_counter))
    print("The accuracy is: {}%".format(round(accuracy,4)))

    fig, ax = plt.subplots()
    ax.scatter(train_x[:,0],train_x[:,1]) #Blue class 1
    ax.scatter(train_y[:,0],train_y[:,1]) #Orange class 2
    ax.scatter(mean_C1[0],mean_C1[1],c="red") #Mean for class 1
    ax.scatter(mean_C2[0],mean_C2[1],c="red") #Mean for class 2
    plt.show()

def task_b():
    """
    Classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy
    """

def task_c():
    """
    classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).
    """

task_a()
