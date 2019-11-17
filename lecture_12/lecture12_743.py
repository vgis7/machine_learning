#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 13/11-2019
# Lecture: 12
# Dependencies  : Algorithm-independent machine learning and reinforcement learning
# Python version: >3.5
# Functionality :
# ###################################
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#Path
import os
current_path = os.getcwd()
os.chdir(current_path)

#Data
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

train_labels = train_data['label']
train_data.drop(columns=['label'])

test_labels = test_data['label']
test_data.drop(columns=['label'])

#Default AdaBoost
AdaBoost = AdaBoostClassifier(n_estimators=100)
AdaBoost.fit(train_data,train_labels)

score = AdaBoost.score(test_data,test_labels)
print(f"Default AdaBoost Accuracy: {score}")


#SVC AdaBoost
svc=SVC(probability=True, kernel='poly')

AdaBoostSVC = AdaBoostClassifier(n_estimators=1,base_estimator=svc)
AdaBoostSVC.fit(train_data,train_labels)

print(f"SVC AdaBoost Accuracy: {score}")
