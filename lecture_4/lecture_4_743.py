#!.venv/bin/python3
#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 18/9-2019
# Lecture: 4  Dimensionality reduction
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality : This script uses PCA to reduce the 728 dimensional input data-set 
#                 to 2 dimensions, after which it uses logistic regression to classify the test data
# ###################################

#5,6,8 <--
# Do exercise2:  (1) from the 10-class database, choose three classes (5, 6 and 8)
# and then reduce dimension to 2; (2) perform 3-class classification based on the
# generated 2-dimensional data

# Mat file (train0-9 and test 0-9)

#__________________________LIBRARIES______________________________#
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#________________________DATA COLLECTION__________________________#
print("[INFO]: Setting up data...")
def collect_labels(dataframe,mat_array,label): #dataframe = dataframe that contains all labels, mat_array = array from the mat file, label = is the label of the digit within the mat file
  temp_class_df = pd.DataFrame(columns=['Label']) #Temporary utilized for storing labels

  for i in tqdm(range(len(mat_array)),desc=f'Collecting labels from mat_file[train{str(label)}]'): #Goes through as many indices as the length of the mat array
    temp_class_df.at[i,'Label'] = label #Sets the label to the specific index, in column 'Label' of the temporary dataframe of labels

  dataframe = dataframe.append(temp_class_df,ignore_index=True) #The temporary dataset of labels is appended upon the dataset that contains all labels
  return dataframe #Returns the dataset of labels

mat_file = loadmat("data/mnist_all.mat") #Loading the mat file which includes the MNIST dataset
train_labels_df = pd.DataFrame(columns=['Label']) #Declares pandas dataframe which contains train labels
train_labels_df = collect_labels(train_labels_df,mat_file['train5'],5) #Collects labels for each digit 5 image and saves it in the train label dataframe
train_labels_df = collect_labels(train_labels_df,mat_file['train6'],6) #Collects labels for each digit 6 image and saves it in the train label dataframe
train_labels_df = collect_labels(train_labels_df,mat_file['train8'],8) #Collects labels for each digit 8 image and saves it in the train label dataframe

test_labels_df = pd.DataFrame(columns=['Label']) #Declares pandas dataframe which contains test labels
test_labels_df = collect_labels(test_labels_df,mat_file['test5'],5) #Collects labels for each digit 8 image and saves it in the test label dataframe
test_labels_df = collect_labels(test_labels_df,mat_file['test6'],6) #Collects labels for each digit 8 image and saves it in the test label dataframe
test_labels_df = collect_labels(test_labels_df,mat_file['test8'],8) #Collects labels for each digit 8 image and saves it in the test label dataframe

#_______________SETUP DATASETS FOR TRAIN AND TEST__________________#
train = np.concatenate((mat_file['train5'],mat_file['train6'],mat_file['train8'])) #Adds all three digits to one numpy array for the train dataset
test = np.concatenate((mat_file['test5'],mat_file['test6'],mat_file['test8'])) #Adds all three digits to one numpy array for the test dataset

#________________________PRE-PROCESSING____________________________#
print("[INFO]: Pre-Processing...")
scaler = StandardScaler() #Used for standardizing features by removing scale and mean to unit variance
scaler.fit(train.astype(float)) #Computes mean and std which is only done using the train dataset
train = scaler.transform(train.astype(float)) #Transforms the train dataset by utilzing the fited scaler
test = scaler.transform(test.astype(float)) #Transforms the train dataset by utilzing the fited scaler

#_____PRINCIPAL COMPONENT ANALYSIS (DIMENSIONALITY REDUCTION)______#
print("[INFO]: Dimensionality Reduction...")
pca = PCA(n_components=2) #Reduces the dimensionality to components of 2
pca.fit(train) #Fits the pca model to the train dataset
train = pca.transform(train) #Applies dimentionality reduction to train dataset
test = pca.transform(test) #Applies dimentionality reduction to test dataset

print("[INFO]: Classificationn...")
#______________LOGISTIC REGRESSION (CLASSIFICATION)________________#
logisticRegr = LogisticRegression(solver='lbfgs',multi_class='multinomial') #Used for classification
logisticRegr.fit(train,train_labels_df['Label'].astype('int')) #Fits the logistic regression model
predictions = logisticRegr.predict(test) #Predicts labels of the test dataset

#__________________CALCULATE MODEL ACCURACY________________________#
print("[INFO]: Calculate Model Accuracy...")
def calculate_accuracy(predictions,labels_df): #Calculates the accuracy of the obtained predictions, using the known labels.
  correct_predictions = 0 #Counts the number of accurate predictions
  for idx,prediction in enumerate(predictions): #Goes through each prediction
    if prediction == labels_df['Label'][idx]: #Checks if the predicted label is equivalent to the true label
      correct_predictions += 1 #Increments if correct
  accuracy = round(((correct_predictions/len(predictions))*100),3) #Calculates the accuracy of correct predictions compared to the number of predictions.
  print(f"[MODEL]: The model accuracy is: {accuracy}%") #Outputs the accuracy

calculate_accuracy(predictions,test_labels_df) #Calls the function.

#_________________________SCATTER PLOT_____________________________#
print("[INFO]: Plotting...")
fig, ax = plt.subplots()
index_5_begin = train_labels_df.loc[train_labels_df['Label']==5.0].index[0] #Locates the first index of the label 5 in the columns 'label' in the dataframe of labels. This enables the possibility of plotting the x,y positions of the dimentional reduced digits of 5
index_6_begin = train_labels_df.loc[train_labels_df['Label']==6.0].index[0] #Locates the first index of the label 6 in the columns 'label' in the dataframe of labels. This enables the possibility of plotting the x,y positions of the dimentional reduced digits of 6
index_8_begin = train_labels_df.loc[train_labels_df['Label']==8.0].index[0] #Locates the first index of the label 8 in the columns 'label' in the dataframe of labels. This enables the possibility of plotting the x,y positions of the dimentional reduced digits of 8

#Digit 5
ax.scatter(train[index_5_begin:index_6_begin,0],
           train[index_5_begin:index_6_begin,1],
           alpha=0.1,c='b',marker='$5$'
           )

#Digit 6
ax.scatter(train[index_6_begin:index_8_begin,0],
           train[index_6_begin:index_8_begin,1],
           alpha=0.1,c='g',marker='$6$'
           )

#Digit 8
ax.scatter(train[index_8_begin::,0],
           train[index_8_begin::,1],
           alpha=0.1,c='r',marker='$8$'
           )

plt.show()
