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



train_data = pd.read_csv("./data/mnist_train.csv")
test_data = pd.read_csv("./data/mnist_test.csv")

train_labels = train_data['label']
train_data.drop(columns=['label'])

test_labels = test_data['label']
test_data.drop(columns=['label'])
