#  ###################################
# Group ID : 743
# Members : Mathias Stougaard Lynge, Moaaz allahham, Kasper Schøn Henriksen, Mikkel D.B. Jeppesen
# Date : 23/10-2019
# Lecture: 8  MLP
# Dependencies  : Are all described in requirements.txt
# Python version: >3.5
# Functionality :
# ###################################

from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pandas as pd

#Setup parameters
EPOCHS = 200
HIDDEN_LAYERS = 30

#LDA 9 dimentions
mnist_lda_data = loadmat('lecture_8/mnist_lda.mat')
test_data = mnist_lda_data['test_data']
train_data = mnist_lda_data['train_data']

train_class = mnist_lda_data['train_class']
test_class = mnist_lda_data['test_class']

mlpLDA = MLPClassifier(max_iter=EPOCHS,hidden_layer_sizes=(HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS),random_state=1,verbose=1)
mlpLDA.fit(train_data,train_class)

score = mlpLDA.score(test_data,test_class)
print(f'LDA score is {score}')

#PCA 15 dimentions
pca_train_data = pd.read_csv("data/mnist_train.csv")
pca_train_Labels = pca_train_data['label']
pca_train_data.drop(columns=['label'])

#Load dataset test
pca_test_data = pd.read_csv("data/mnist_test.csv")
pca_test_labels = pca_test_data['label']
pca_test_data.drop(columns=['label'])

for i in range(1,4):
    mlpPCA = MLPClassifier(max_iter=EPOCHS,hidden_layer_sizes=(HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS,HIDDEN_LAYERS),random_state=1,verbose=1)

    component_number = 10*i
    pca = PCA(n_components = component_number)
    pcaReducedTrain = pca.fit_transform(pca_train_data)
    pcaReducedTest = pca.transform(pca_test_data)

    mlpPCA.fit(pcaReducedTrain,pca_train_Labels)
    score = mlpPCA.score(pcaReducedTest,pca_test_labels)
    print(f'PCA {component_number} score is {score}')