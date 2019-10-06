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
import matplotlib.pyplot as plt

#Load dataset train
train = pd.read_csv("mnist_train.csv")
trainLabels = train['label']
train.drop(columns=['label'])

#Load dataset test
test = pd.read_csv("mnist_test.csv")
testLabels = test['label']
test.drop(columns=['label'])

maximum_components = 10 #Goes from 2 to maximum_components for both LDA and PCA.

def PredictWithLDA():
    LDA_scores = []
    for i in range(2,maximum_components):

        #Creates lda
        lda = LDA(n_components = i)

        #Fits and transforms the dataset by use of LDA
        ReducedTrainLDA = lda.fit_transform(train,trainLabels)
        ReducedTestLDA = lda.transform(test)

        #Creates LinearRegression for to be used for the LDA
        lrLDA = LinearRegression()

        #Fits data with Linear Regression:
        lrLDA.fit(ReducedTrainLDA,trainLabels)

        #Predicts and checks the evaluates according to the labels
        score1 = lrLDA.score(ReducedTestLDA,testLabels)
        print(f"LDA score with {i} components: " + str(score1))
        LDA_scores.append(score1)
    return LDA_scores

def PredictWithPCA():
    PCA_scores = []
    for i in range(2,maximum_components):

        #Creates LinearRegression for to be used for the PCA
        lrPCA = LinearRegression()


        #Reduce data by PCA:
        pca = PCA(n_components = i)
        trainReducedPCA = pca.fit_transform(train)
        testReducedPCA = pca.transform(test)

        #Fits data with Linear Regression:
        lrPCA.fit(trainReducedPCA, trainLabels)

        #Predicts and checks the evaluates according to the labels
        score2 = lrPCA.score(testReducedPCA, testLabels)
        print(f"PCA score with {i} components: "+ str(score2))
        PCA_scores.append(score2)
    return PCA_scores

LDA_scores = PredictWithLDA()
PCA_scores = PredictWithPCA()

plt.scatter(range(2,maximum_components),LDA_scores,c = 'r')
plt.plot(range(2,maximum_components),LDA_scores)
plt.scatter(range(2,maximum_components),PCA_scores,c = "b")
plt.plot(range(2,maximum_components),PCA_scores)
plt.show()
