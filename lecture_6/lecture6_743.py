#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 02/10-2019
# Lecture: 6  Linear discrimination
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality :
# ###################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

#Load dataset train
train = pd.read_csv("../mnist_train.csv")
trainLabels = train['label']
train.drop(columns=['label'])

#Load dataset test
test = pd.read_csv("mnist_test.csv")
testLabels = test['label']
test.drop(columns=['label'])

#Creates lda
lda = LDA(n_components = 9)

#Fits and transforms the dataset by use of LDA
ReducedTrainLDA = lda.fit_transform(train,trainLabels)
ReducedTestLDA = lda.transform(test)

#Creates LinearRegression for to be used for the LDA
lrLDA = LinearRegression()

#Fits data with Linear Regression:
lrLDA.fit(ReducedTrainLDA,trainLabels)

#Predicts and checks the evaluates according to the labels
score1 = lrLDA.score(ReducedTestLDA,testLabels)
print("LDA score: " + str(score1))

#Creates LinearRegression for to be used for the PCA
lrPCA = LinearRegression()


#Reduce data by PCA:
pca = PCA(n_components = 9)
trainReducedPCA = pca.fit_transform(train)
testReducedPCA = pca.transform(test)

#Fits data with Linear Regression:
lrPCA.fit(trainReducedPCA, trainLabels)

#Predicts and checks the evaluates according to the labels
score2 = lrPCA.score(testReducedPCA, testLabels)
print("PCA score: "+ str(score2))
