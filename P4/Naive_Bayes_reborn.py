from mldata import parse_c45
from mldata import ExampleSet
from math import log2
import numpy as np
import math


def splitDataset(data_newarray, rows):
    cv_list = [[],[],[]]
    chunk_size = math.floor(rows/3)
    cv_list[0] = data_newarray[ 0:chunk_size ,:]
    cv_list[1] = data_newarray[ chunk_size:2*chunk_size ,:]
    cv_list[2] = data_newarray[ 2*chunk_size:-1 ,:]
    return cv_list

def seperateValidationSet( data_list, n):
    if n == 0:
        validationSet = data_list[0]
        trainSet = np.vstack((data_list[1],data_list[2]))
    elif n == 1:
        validationSet = data_list[1]
        trainSet = np.vstack((data_list[0],data_list[2]))
    elif n == 2:
        validationSet = data_list[2]
        trainSet = np.vstack((data_list[0],data_list[1]))
    else:
        print("-----Error: N is more than 3 fold -----")
    return  trainSet, validationSet

def makeUnchosenColumns(columns, chosen_columns):
    whole_column = list(range(1,columns))
    if chosen_columns == None:
        return whole_column
    for i in chosen_columns:
        whole_column.remove(i)
    return whole_column

def features_prob(maxminarray,save_continuous,data_list):
    saveTrainSet = [[],[],[]]
    saveValidationSet = [[],[],[]]
    saveValueRatio = [[],[],[]] # 4 dimension list

    for nValidation in range(2):
        trainSet, validationSet =  seperateValidationSet(data_list, nValidation)
        saveTrainSet[nValidation] = trainSet
        saveValidationSet[nValidation] = validationSet
        valueRatio = conditionalProb(trainSet, maxminarray, save_continuous)
        saveValueRatio[nValidation] = valueRatio

    return saveTrainSet,saveValidationSet,saveValueRatio

def conditionalProb(trainSet, maxminarray, save_continuous):

    valueRatio = [] # 3 dimension list, element are 2 dimension array:
    '''
    feature: 
    array([ value1, p(x=i|y=1), p(x=i|y=0)
            value2, p(x=i|y=1), p(x=i|y=0)
            value3, p(x=i|y=1), p(x=i|y=0)
            ])
    '''   
    rows, columns = np.shape(trainSet)
    trainsetLabel = trainSet[:,-1]
    if columns != np.shape(maxminarray)[1]+2:
        print("Number of columns not match!")

    for nColumns in range(1, columns-1):
        if nColumns in save_continuous:
            trainSet[:,nColumns] = thresholdContinuous(trainSet[:,nColumns], maxminarray[:,nColumns-1],rows)
        featuresRatio = calculateConditonalProb(trainSet[:,nColumns],trainsetLabel)
        valueRatio.append([featuresRatio])
        #[ [featureRatio1], [featureRatio2], [featureRatio3] ... ]

    return valueRatio

def thresholdContinuous(continuousFeature, maxminFeature, rows):
    minValue = maxminFeature[0]
    maxValue = maxminFeature[1]
    # if np.max(continuousFeature) != maxValue:
    #     print("Max continuousFeature is ", continuousFeature, "MaxValue is :", maxValue )
    featureRange = maxValue - minValue
    threshold_1 = featureRange/3 + minValue
    threshold_2 = featureRange/3*2 + minValue
    # sepereate continuous value into 3 parts

    for nRows in range(rows):
        if continuousFeature[nRows] < threshold_1:
            continuousFeature[nRows] = 0
        elif continuousFeature[nRows] < threshold_2:
            continuousFeature[nRows] = 1
        elif continuousFeature[nRows] < maxValue or continuousFeature[nRows] == maxValue:
            continuousFeature[nRows] = 2
        else: 
            print(continuousFeature[nRows])
            print("continous value is out of range")
    return continuousFeature

def calculateConditonalProb( discreteFeature , trainsetLabel):
    
    temp = 0
    combinedArray = np.transpose(np.vstack(( discreteFeature, trainsetLabel )) )
    sortCombineArray = combinedArray[combinedArray[:,0].argsort()]

    y = np.bincount(discreteFeature.astype(int))
    ii = np.nonzero(y)[0]
    value_count = np.vstack((ii,y[ii])).T # each value with counts
    valueNum = value_count[:,1]
    valueLen = len(valueNum)
    valueNum = value_count[:,1]
    featuresRatio = np.zeros([valueLen, 3])
    featuresCount = np.zeros([valueLen, 2])

    for nValueLen in range(valueLen):
        # choose one value
        featuresRatio[nValueLen, 0] = valueNum[nValueLen]
        for i in range(temp, value_count[nValueLen, 1]+temp):
            # count pos and neg labels  
            if sortCombineArray[nValueLen, 1] == 1:
                featuresCount[nValueLen,0] += 1
            elif sortCombineArray[nValueLen, 1] == 0:
                featuresCount[nValueLen,1] += 1
        temp = i
    featuresCount[:,0] = featuresCount[:,0] / sum(featuresCount[:,0])
    featuresCount[:,1] = featuresCount[:,1] / sum(featuresCount[:,1])
    featuresRatio[:,1:3] = featuresCount
    # featuresRatio:
    # array([value1, p(v=vi|y=1),p(v=vi|y=0)]
    # [value2, p(v=vi|y=1),p(v=vi|y=0)]
    # [value3, p(v=vi|y=1),p(v=vi|y=0)])

    return featuresRatio


def makeUnchosenColumns(columns, chosen_columns):
    whole_column = list(range(1,columns))
    if chosen_columns == None:
        return whole_column
    for i in chosen_columns:
        whole_column.remove(i)
    return whole_column


# chosen columns should range(1, cols-1)
def cvErrorRate(nValidation, saveValidationSet, saveValueRatio, chosen_features, attempFeature):
    # trainSet = saveTrainSet[nValidation]
    validationSet = saveValidationSet[nValidation]
    valueRatio = saveValueRatio[nValidation]
    validationSetRows , validationSetCols  = np.shape(validationSet)
    validationLabels = validationSet[:,-1]
    labelRatio = calLabelRatio(validationLabels)
    chosen_features.append(attempFeature)
    eRateNumerator = 0
    eRateDenominator = 0

    for nValidationSetRows in range(validationSetRows):
        oneExample = validationSet[nValidationSetRows , :]
        saveExampleProb = conditionalProbArray(oneExample,valueRatio,chosen_features)
        saveExampleProb = np.vstack(( labelRatio, saveExampleProb ))
        predictLabel = naive_calculate(saveExampleProb)
        if( predictLabel != validationLabels[nValidationSetRows] ):
            eRateNumerator += 1
        eRateDenominator += 1
    eRate = eRateNumerator/ eRateDenominator

    return eRate



def calLabelRatio(labels):
    labels_1_ind = np.where(labels == 1)
    labels_0_ind = np.where(labels == 0)
    labels_1_ratio = len(labels_1_ind)/(len(labels_1_ind)+len(labels_0_ind))
    labels_0_ratio = len(labels_0_ind)/(len(labels_1_ind)+len(labels_0_ind))
    labelRatio = np.array([labels_1_ratio, labels_0_ratio])
    return labelRatio

def conditionalProbArray(oneExample,valueRatio,chosen_features):
    # oneExample columns: (1,cols-1) len: cols-2
    #    chosen_features is based on this range
    # valueRatio columns: (0,cols-2) len: cols-2
    saveExampleProb = []
    for nChosen_features in chosen_features:
        features_ratio = valueRatio[nChosen_features-1]
        features_example = oneExample[chosen_features]
        flagSite = np.where( (features_ratio[0])[:,0] == features_example[0] )
        # find all prob ratio for examples , flagSite:(array([?]),) 
        flag = (flagSite[0])
        exampleProb = features_ratio[flag,1:2]
        saveExampleProb.append([exampleProb])
    return saveExampleProb

    
def naive_calculate(final_array):
    # print(final_array)
    # if np.size(final_array) == 2:
    #     p_x_1 = final_array[0]
    #     p_x_0 = final_array[1]
    # else:
    prob_pos = [x[0] for x in final_array]
    prob_neg = [x[1] for x in final_array]
    p_x_1 = np.prod(prob_pos)
    p_x_0 = np.prod(prob_neg)

    if (p_x_1 > p_x_0) :
        return 1
    elif (p_x_1 < p_x_0):
        return 0
    else: 
        return  np.random.randint(2,size=1)



        

