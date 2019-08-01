from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import pandas as pd
import random

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False
length = 0
# a function for shuffling
def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

# takes the sameFile and diffFile, take the same amount of data pairs and shuffle together
# return a 2D array: SuffledList[][]
def GenerateShuffledData(sameFile, diffFile):
    temp1 = []
    temp2 = []
    temp3 = []
    file1 = []
    file2 = []
    file3 = []
    with open(sameFile, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            temp1.append(row[0])
            temp2.append(row[1])
            temp3.append(row[2])
        for index in range(1,len(temp1)):
            file1.append(temp1[index])
            file2.append(temp2[index])
            file3.append(int(temp3[index]))

    temp4 = []
    temp5 = []
    temp6 = []
    file4 = []
    file5 = []
    file6 = []
    with open(diffFile, 'rU') as f:
        reader1 = csv.reader(f)
        for row in reader1:
            temp4.append(row[0])
            temp5.append(row[1])
            temp6.append(row[2])
        for index in range(1,len(temp1)):
            file4.append(temp4[index])
            file5.append(temp5[index])
            file6.append(int(temp6[index]))

    mixfileColumn1 = []
    mixfileColumn2 = []
    mixfileColumn3 = []
    for index in range(0,len(file1)):
        mixfileColumn1.append(file1[index])
        mixfileColumn1.append(file4[index])
    for index in range(0,len(file1)):
        mixfileColumn2.append(file2[index])
        mixfileColumn2.append(file5[index])
    for index in range(0,len(file1)):
        mixfileColumn3.append(file3[index])
        mixfileColumn3.append(file6[index])

    a, b, c = shuffle_list(mixfileColumn1, mixfileColumn2, mixfileColumn3)
    w, h = 3, len(a)
    SuffledList = [[0 for x in range(w)] for y in range(h)]
    for index_Y in range(0, len(a)):
        SuffledList[index_Y][0] = a[index_Y]
        SuffledList[index_Y][1] = b[index_Y]
        SuffledList[index_Y][2] = c[index_Y]

    return SuffledList

# This function reads a .csv file and output the file data in an t[] array
def GetTargetVector(filePath):
    t = []
    for index in range(0, len(filePath)):
        t.append(filePath[index][2])
    return t

# Generate the Human Observed Rowdata into a dictionary
def GenerateHOFDdictionary(FeatureDataFile):
    with open(FeatureDataFile, 'rU') as f:
        reader = csv.reader(f)
        dic = {(row[1]):row[2:] for row in reader}
    return dic

# Generate the GSC Rowdata into a dictionary
def GenerateGSCdictionary(FeatureDataFile):
    with open(FeatureDataFile, 'rU') as f:
        reader = csv.reader(f)
        dic = {(row[0]):row[1:] for row in reader}
    return dic

# Concatenation
def FeatureConcatenation(FeatureData, TargetVector):
    dataMatrix = []
    for index in range(0, len(TargetVector)):
        Feature_a = FeatureData.get(TargetVector[index][0])
        Feature_b = FeatureData.get(TargetVector[index][1])
        dataMatrix.append(Feature_a+Feature_b)
    dataMatrix = np.array(dataMatrix).astype(np.int)
    dataMatrix = np.transpose(dataMatrix)
    return dataMatrix

# Substruction
def FeatureSubtraction(FeatureData, TargetVector):
    dataMatrix = []
    for index in range(0, len(TargetVector)):
        Feature_a = FeatureData.get(TargetVector[index][0])
        Feature_b = FeatureData.get(TargetVector[index][1])

        sub = []
        for FeatureLen in range(0, len(Feature_a)):

            sub.append(np.abs(int(Feature_a[FeatureLen]) - int(Feature_b[FeatureLen])))
        dataMatrix.append(sub)
    dataMatrix = np.transpose(dataMatrix)
    return dataMatrix

# Generating the Training Target: 80% of total
def GenerateTrainingTarget(rawTraining, TrainingPercent=80):
    TrainingLen = int(math.ceil(len(rawTraining) * (TrainingPercent * 0.01)))
    t = rawTraining[:TrainingLen]
    # print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# Generating the Training Data: 80% of total
def GenerateTrainingDataMatrix(rawData, TrainingPercent=80):
    T_len = int(math.ceil(len(rawData[0]) * 0.01 * TrainingPercent))
    d2 = rawData[:, 0:T_len]
    # print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# Generating Validation Data: 10% of total
def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0]) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:, TrainingCount + 1:V_End]
    # print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

# Generating Validation Target Data: 10% of total
def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount + 1:V_End]
    # print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Generating Big Sigma for later calculating
def GenerateBigSigma(Data, MuMatrix, TrainingPercent, IsSynthetic):
    BigSigma = np.zeros((len(Data), len(Data)))
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    varVect = []
    for i in range(0, len(DataT[0])):
        vct = []
        for j in range(0, int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3, BigSigma)
    else:
        BigSigma = np.dot(200, BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


def GetScalar(DataRow, MuRow, BigSigInv):
    R = np.subtract(DataRow, MuRow)
    T = np.dot(BigSigInv, np.transpose(R))
    L = np.dot(R, T)
    return L


def GetRadialBasisOut(DataRow, MuRow, BigSigInv):
    phi_x = math.exp(-0.5 * GetScalar(DataRow, MuRow, BigSigInv))
    return phi_x

# Generating the design Matrix
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent=80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    PHI = np.zeros((int(TrainingLen), len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for C in range(0, len(MuMatrix)):
        for R in range(0, int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    # print ("PHI Generated..")
    return PHI

# Calculating the weight
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0, len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T = np.transpose(PHI)
    PHI_SQR = np.dot(PHI_T, PHI)
    PHI_SQR_LI = np.add(Lambda_I, PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER = np.dot(PHI_SQR_INV, PHI_T)
    W = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# Same function as the above, generating the design Matrix
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent=80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    PHI = np.zeros((int(TrainingLen), len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for C in range(0, len(MuMatrix)):
        for R in range(0, int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    # print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI, W):
    Y = np.dot(W, np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# Calculating the accuracy and Erms
# put the accuracy and Erms in an array at index 0 and 1, seperating by ','
def GetErms(VAL_TEST_OUT, ValDataAct):
    sum = 0.0
    t = 0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range(0, len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]), 2)
        if (int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter += 1
    accuracy = (float((counter * 100)) / float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' + str(math.sqrt(sum / len(VAL_TEST_OUT))))

# Blow function is for logistic regression-------------------------------------------------

# getting the accuracy for compare the predict[i] and target data[i]
# rewrite the GetErms() so that i can add Weight to the parameter
def GetErmsLogistic(VAL_TEST_OUT, ValDataAct, Weight):
    sum = 0.0
    accuracy = 0.0
    counter = 0

    pred = decisionBoundary(predict(VAL_TEST_OUT, Weight))

    for i in range(0, len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]), 2)
        if (pred[i] == ValDataAct[i]):
            counter += 1
    accuracy = (float((counter * 100)) / float(len(VAL_TEST_OUT)))

    return (str(accuracy) + ',' + str(math.sqrt(sum / len(VAL_TEST_OUT))))

# calculating the predict: Rowdata*Weights
def predict(features, weights):
    print(features.shape)
    print(weights.shape)
    z = np.dot(features, weights)
    return 1/(1+np.exp(-z))

# calculating the cost
def costFunction(features, labels, weights):
    observations = len(labels)
    predictions = predict(features, weights)
    ErrorEqual1 = -labels*np.log(predictions)
    ErrorEqual0 = (1-labels)*np.log(1-predictions)
    cost = ErrorEqual1 - ErrorEqual0
    cost = cost.sum()/observations
    return cost

# calculating the Weight, return the final weights
def updateWeights(features, labels, weights, learningRate):
    N = len(features)

    # Get prediction
    predictions = predict(features.T, weights)
    gradient = np.dot(features, predictions-labels)
    gradient /= N
    gradient *= learningRate
    weights -= gradient

    return weights

# classify the probability into 1 or 0
def decisionBoundary(prob):
    return 1 if prob >= 0.5 else 0






# -------------------------------------------------------Main--------------------------------------------------------

# Data process
HOFDFile = GenerateShuffledData('same_pairs.csv','diffn_pairs.csv')
HOFDdic = GenerateHOFDdictionary('HumanObserved-Features-Data.csv')

# Human observed Feature target value
HOFDRawTarget = GetTargetVector(HOFDFile)

# Human observed Feature row data
#HOFDAddRowData = FeatureConcatenation(HOFDdic,HOFDFile)
HOFDSubRowData = FeatureSubtraction(HOFDdic,HOFDFile)


# GSCFile = GenerateShuffledData('GSC_same_pairs.csv','GSC_diffn_pairs.csv')
# GSCdic = GenerateGSCdictionary('GSC-Features.csv')
# GSCRawTarget = GetTargetVector(GSCFile)
#
## GSC Feature target value
# GSCRawTarget = GetTargetVector(GSCFile)

## GSC Feature row data
# #GSCAddRowData = FeatureConcatenation(GSCdic,GSCFile)
# GSCSubRowData = FeatureSubtraction(GSCdic,GSCFile)

RawTarget = HOFDRawTarget
RawData   = HOFDSubRowData

# print("Row data shape is ")
# print(RawTarget.shape)
# print(RawTarget.shape)
# print(RawTarget.shape)

TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)

ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)

TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')



# -------------------------------------Logistic Regression-----------------------------------------------------
#
#
#
# La = 2
# learningRate = 0.01
# Lg_Acc_Val = []
# Lg_Acc_TR = []
# Lg_Acc_Test = []
# temp_List = []
#
# List_TR = []
# List_Val = []
# List_Test = []
#
#
#
#
# print ('----------------------------------------------------')
# print ('--------------Please Wait for 2 mins!----------------')
# print ('----------------------------------------------------')
#
#
# #print(TrainingData)
#
# W_Now = np.zeros(9)
#
# print("test")
# print(HOFDSubRowData.shape)
# print(TrainingData)
# TrainingData = np.array(TrainingData)
# TrainingData = np.transpose(TrainingData)
# print(TrainingData)
# print(TrainingData.shape)
#
# print("Test ENDIng")
#
# #for L_index in range(1,2):
#     #temp_List.append(str(L_index))
# #TrainingData = np.transpose(TrainingData)
# #TrainingData = np.array(TrainingData)
#
# costHistory = []
# for index in range(0,400):
#     pred = predict(TrainingData,W_Now)
#     gradient = np.dot(TrainingData.T, pred-TrainingTarget)
#     gradient = gradient/len(TrainingTarget)
#     W_Now = W_Now - (learningRate*gradient)
#
#
#     #W_Now = updateWeights(TrainingData, TrainingTarget, W_Now, learningRate)
#     cost = costFunction(TrainingData, TrainingTarget, W_Now)
#     costHistory.append(cost)
#
#     # -----------------TrainingData Accuracy---------------------#
#     Erms_TR = GetErmsLogistic(TrainingData, TrainingTarget, W_Now)
#     Lg_Acc_TR.append(float(Erms_TR.split(',')[0]))
#
#
#     # -----------------ValidationData Accuracy---------------------#
#     Erms_Val = GetErmsLogistic(ValData, ValDataAct, W_Now)
#     Lg_Acc_Val.append(float(Erms_Val.split(',')[0]))
#
#
#      # -----------------TestingData Accuracy---------------------#
#     Erms_Test = GetErmsLogistic(TestData, TestDataAct, W_Now)
#     Lg_Acc_Test.append(float(Erms_Test.split(',')[0]))
#
# List_TR.append(str(np.around(min(Lg_Acc_TR), 5)))
# List_Val.append(str(np.around(min(Lg_Acc_Val), 5)))
# List_Test.append(str(np.around(min(Lg_Acc_Test), 5)))
#
#
# # a code ised for data collection stored in .cvs
# df = pd.DataFrame(data={"K-mean": temp_List, "Training Accuracy": List_TR, "Validation Accuracy": List_Val, "Testing Accuracy": List_Test})
# df.to_csv("Logistic_Regression_List.csv", sep=',',index = False)



# -------------------------------------Stochastic Gradient Decent-----------------------------------------------------





La = 2
learningRate = 0.01
L_Erms_Val = []
L_Erms_TR = []
L_Erms_Test = []
L_Acc_Val = []
L_Acc_TR = []
L_Acc_Test = []
L_Cluster = []
List_ETR_TR = []
List_ETR_Val = []
List_ETR_Test = []
List_Acc_TR = []
List_Acc_Val = []
List_Acc_Test = []
W_Mat = []
# I use an for loop to help me better collect all the data
for L_index in range(1,11):



    #La = L_index
    #learningRate=L_index/100
    iteration = L_index

    kmeans = KMeans(n_clusters=10, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_

    # Calculating the elements
    BigSigma = GenerateBigSigma(RawData, Mu, TrainingPercent, IsSynthetic)

    for index in range(0, len(BigSigma)):
        if BigSigma[index][index] == 0:
            BigSigma[index][index] = 0.01

    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    W = GetWeightsClosedForm(TRAINING_PHI, TrainingTarget, (C_Lambda))
    TEST_PHI = GetPhiMatrix(TestData, Mu, BigSigma, 100)
    VAL_PHI = GetPhiMatrix(ValData, Mu, BigSigma, 100)

    L_Cluster.append(str(L_index))

    W_Now = np.dot(220, W)

    #Training data with iterations, every iteration it updates its weights and biases.
    for i in range(0, iteration):
                # print ('---------Iteration: ' + str(i) + '--------------')
                Delta_E_D = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now), TRAINING_PHI[i])), TRAINING_PHI[i])
                La_Delta_E_W = np.dot(La, W_Now)
                Delta_E = np.add(Delta_E_D, La_Delta_E_W)
                Delta_W = -np.dot(learningRate, Delta_E)
                W_T_Next = W_Now + Delta_W
                W_Now = W_T_Next

                # -----------------TrainingData Accuracy---------------------#
                TR_TEST_OUT = GetValTest(TRAINING_PHI, W_T_Next)
                Erms_TR = GetErms(TR_TEST_OUT, TrainingTarget)
                L_Erms_TR.append(float(Erms_TR.split(',')[1]))
                L_Acc_TR.append(float(Erms_TR.split(',')[0]))


                # -----------------ValidationData Accuracy---------------------#
                VAL_TEST_OUT = GetValTest(VAL_PHI, W_T_Next)
                Erms_Val = GetErms(VAL_TEST_OUT, ValDataAct)
                L_Erms_Val.append(float(Erms_Val.split(',')[1]))
                L_Acc_Val.append(float(Erms_Val.split(',')[0]))


                # -----------------TestingData Accuracy---------------------#
    TEST_OUT = GetValTest(TEST_PHI, W_T_Next)
    Erms_Test = GetErms(TEST_OUT, TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    L_Acc_Test.append(float(Erms_Test.split(',')[0]))
    print("im done")
    print(L_index)

    List_ETR_TR.append(str(np.around(min(L_Erms_TR), 5)))
    List_Acc_TR.append(str(L_Acc_TR[-1]))
    List_ETR_Val.append(str(np.around(min(L_Erms_Val), 5)))
    List_Acc_Val.append(str(L_Acc_Val[-1]))
    List_ETR_Test.append(str(np.around(min(L_Erms_Test), 5)))
    List_Acc_Test.append(str(L_Acc_Test[-1]))

for z in range(0,len(L_Cluster)):

    print('----------Gradient Descent Solution--------------------')
    print ("M = {}".format(10) + "\nLambda  = {}".format(La) + "\neta={}".format(learningRate))
    print ("E_rms Training   = " + str(List_ETR_TR[z]))
    print ("E_rms Validation = " + str(List_ETR_Val[z]))
    print ("E_rms Testing    = " + str(List_ETR_Test[z]))
    print ("Accuracy Training   = " + str(List_Acc_TR[z]))
    print ("Accuracy Validation   = " + str(List_Acc_Val[z]))
    print ("Accuracy Testing   = " + str(List_Acc_Test[z]))


# a code ised for data collection stored in .cvs
df = pd.DataFrame(data={"Iteration": L_Cluster, "Training Accuracy": List_Acc_TR, "Validation Accuracy": List_Acc_Val, "Testing Accuracy": List_Acc_Test,"E_training": List_ETR_TR, "E_validation": List_ETR_Val, "E_testing": List_ETR_Test})
df.to_csv("Sample_L_List.csv", sep=',',index = False)
