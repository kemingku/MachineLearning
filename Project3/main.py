from sklearn.cluster import KMeans

import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import pandas as pd
import random
import pickle
import gzip
from PIL import Image
import os
import scipy.sparse


import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata


def predictHypothesis(x, weight):
    z = np.dot(np.transpose(weight), x)
    predict = np.exp(z)/sum

# Load MNIST on python
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
trainMat = training_data[0]
trainTar = training_data[1]
MNISTvalMat = test_data[0]
MNISTvalTar = test_data[1]
#print(training_data)
#print(len(training_data[0]))
f.close()


# Load USPS on python
USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):

    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)

#Logistic Regression
print("Softmax Regression------------------------")


y = training_data[1]
x = training_data[0]

trainX = training_data[0]
trainY = training_data[1]
valiX = validation_data[0]
valiY = validation_data[1]
testX = test_data[0]
testY = test_data[1]

uspsX = USPSMat
uspsY = USPSTar

# softmax function
def softmax(z):
    z -= np.max(z)
    result = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return result

# get loss and gradient decent
def getLoss(weight,x,y,la):
    x_temp = x.shape[0]
    y_shape = y.shape[0]
    temp = scipy.sparse.csr_matrix((np.ones(y_shape), (y, np.array(range(y_shape)))))
    y_onehotC = np.array(temp.todense()).T
    scores = np.dot(x,weight)
    prob = softmax(scores)
    loss = (-1 / x_temp) * np.sum(y_onehotC * np.log(prob)) + (la/2)*np.sum(weight*weight)
    grad = (-1 / x_temp) * np.dot(x.T,(y_onehotC - prob)) + la*weight
    return loss,grad

def getProbsAndPreds(someX,weight):
    probs = softmax(np.dot(someX,weight))
    preds = np.argmax(probs,axis=1)
    return probs,preds

# get accuracy
def getAccuracy(someX,someY,weight):
    prob,prede = getProbsAndPreds(someX,weight)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

losses = []
Train_Acc = []
Valiadation_Acc = []
Test_Acc = []
USPS_Acc = []
L_Cluster = []

# using for loop to store all tested result in a csv file
# L_Cluster contain element testing
for L_index in range(10,1001,20):
    print(L_index)
    L_Cluster.append(L_index/100000)
    w = np.zeros([x.shape[1],len(np.unique(y))])
    lam = 1
    iterations = 400
    learningRate = L_index/100000

    for i in range(0,iterations):
        loss,grad = getLoss(w,x,y,lam)
        losses.append(loss)
        w = w - (learningRate * grad)
    print("loss")
    print(loss)
    losses.append(loss)



    print("accuracy")
    Train_Acc.append(getAccuracy(trainX,trainY,w))
    Valiadation_Acc.append(getAccuracy(valiX,valiY,w))
    Test_Acc.append(getAccuracy(testX,testY,w))
    USPS_Acc.append(getAccuracy(uspsX,uspsY,w))
    print(getAccuracy(x,y,w))

df = pd.DataFrame(data={"MNISTLearning Rate": L_Cluster, "MNISTTraining Accuracy": Train_Acc, "MNISTValidation Accuracy": Valiadation_Acc, "MNISTTesting Accuracy": Test_Acc, "USPS Accuracy": USPS_Acc})
df.to_csv("Project4_Sample_List.csv", sep=',',index = False)


# a = np.array([[0.07,0.22,0.28], [0.35,0.78,1.12], [-0.33,-0.58,-0.92], [-0.39,-0.7,-1.1]])
# Tar = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,1]])
# print(a)
# print(Tar)
# predict = (np.exp(a.T)/(sum(np.exp(a.T)))).T
# print(predict)
# print("CrossEntropy")
# print(Tar.shape)
# print(predict.shape)
# Tar.append(Tar)
# print(len(Tar))
# tempTar = []
# tempPre = []
# print(Tar[0])
# for index_x in range(0,len(Tar)):
#     tempTar.append(sum(Tar[index_x]))
# print(tempTar)
# # for index_y in range(0,len(predict)):
# #     temphh = np.multiply(tempTar, (predict[index_y]))
# #
# #     #tempPre.append(sum(np.dot(tempTar.T, (predict[index_y]).T)))
# # print(temphh.T)
#

# CrossEntropy = np.sum(Tar*np.log(predict))
# #CrossEntropy = sum(np.dot(np.transpose(tempTar), np.log(predict)))
# # CrossEntropy = -(sum(np.dot(Tar.T.T,np.log(predict.T)))).T
# print(CrossEntropy.shape)
# print(CrossEntropy.T)

# Neural Network
x_train = trainMat
y_train = trainTar
x_test = USPSMat
y_test = USPSTar
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes=10
image_vector_size=28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
image_size = 784
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False, validation_split=.1)
loss,accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Neural Network accuracy---------------------")
print(accuracy)

# SVM
mnist = fetch_mldata('MNIST original')
print(mnist.data.shape)

n_train = 60000
n_test = 10000
indices = np.arange(len(mnist.data))
train_idx = np.arange(0,n_train)
test_idx = np.arange(n_train+1,n_train+n_test)
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
classifier1 = SVC(kernel='linear');


classifier1.fit(X_train, y_train)
mnistTrain_acc = classifier1.score(X_train, y_train)
mnistTest_acc = classifier1.score(X_test, y_test)
usps_acc = classifier1.score(USPSMat, USPSTar)
print("SVM accuracy----------")
print(mnistTrain_acc)
print(mnistTest_acc)
print(usps_acc)

#RandomForestClassifier
mnist = fetch_mldata('MNIST original')
n_train = 60000
n_test = 10000
indices = np.arange(len(mnist.data))
train_idx = np.arange(0,n_train)
test_idx = np.arange(n_train+1,n_train+n_test)
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

classifier2 = RandomForestClassifier(n_estimators=10);
classifier2.fit(X_train, y_train)
mnistTrain_acc = classifier2.score(X_train, y_train)
mnistTest_acc = classifier2.score(X_test, y_test)
usps_acc = classifier2.score(USPSMat, USPSTar)
print("Random Forest accuracy----------")
print(mnistTrain_acc)
print(mnistTest_acc)
print(usps_acc)