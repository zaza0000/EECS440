from math import log2
import numpy as np
import math
from collections import Counter
from load_data  import *

def make_maxminarray(attr_type_flag, data_value_list):
    numeric_columns = []
    for key in attr_type_flag.keys():
        if attr_type_flag[key] == '1':
            numeric_columns.append(key) # all string

    max_min_dict = {}
    for key in numeric_columns:
        compare_max = []
        for nNumberD in range(len(data_value_list)):
            compare_max.append( float(data_value_list[nNumberD][key]))
        max_min_dict[key] = max(compare_max), min(compare_max)

    return numeric_columns, max_min_dict

def thresholdContinuous(numeric_columns, max_min_dict, numberD, data_value_list, attr_type_flag):
    threshold_dict = {}
    for key in max_min_dict.keys():
        threshold_top, threshold_bot = max_min_dict[key]
        threshold1 = threshold_bot + (threshold_top - threshold_bot)/5
        threshold2 = threshold1 + (threshold_top - threshold_bot)/5
        threshold3 = threshold2 + (threshold_top - threshold_bot)/5
        threshold4 = threshold3 + (threshold_top - threshold_bot)/5
        threshold_dict[key] = [threshold_bot, threshold1, threshold2, threshold3, threshold4, threshold_top]
    for nNumberD in range(numberD):
        for key in data_value_list[nNumberD].keys():
            if attr_type_flag[key] == '1':
                a = float(data_value_list[nNumberD][key])
                if a > threshold_dict[key][0] and threshold_dict[key][1] > a or a == threshold_dict[key][0]:
                    data_value_list[nNumberD][key] = '0'
                elif a > threshold_dict[key][1] and threshold_dict[key][2] > a or a == threshold_dict[key][1]:
                    data_value_list[nNumberD][key] = '1'
                elif a > threshold_dict[key][2] and threshold_dict[key][3] > a or a == threshold_dict[key][2]:
                    data_value_list[nNumberD][key] = '2'
                elif a > threshold_dict[key][3] and threshold_dict[key][4] > a or a == threshold_dict[key][3]:
                    data_value_list[nNumberD][key] = '3'              
                elif a > threshold_dict[key][4] and threshold_dict[key][5] > a or a == threshold_dict[key][4] or a == threshold_dict[key][5]:
                    data_value_list[nNumberD][key] = '4'
    return threshold_dict

def conditional_prob(trans_label, data_value_list,m_estimate):
    conditional_num_dict = {}
    conditional_prob_dict = {}
    conditional_sum_dict = {}
    appeared_key = []
    count_val = {}
    # for key in trans_label.keys():

    for key in trans_label.keys():
        # key means NO.example
        if trans_label[key] in conditional_num_dict.keys():
            conditional_num_dict[ trans_label[key] ].append( data_value_list[int(key)] )
        else:
            conditional_num_dict[ trans_label[key] ] = [ data_value_list[int(key)] ]

    # transform str to float
    conditional_amount_dict = {}
    conditional_prob_dict = {}
    counter = 0
    save = 0
    for key in conditional_num_dict.keys():             #  8 labels 
        conditional_prob_dict[key] = {}
        for key2 in conditional_num_dict[key][0].keys(): # 103 attributes
            conditional_prob_dict[key][key2] = {}
            for nLen in range( len(conditional_num_dict[key]) ):  # number of example under each label
                the_val = conditional_num_dict[key][nLen][key2]
                count_val[the_val] = 1
                if the_val in conditional_prob_dict[key][key2]:
                    conditional_prob_dict[key][key2][the_val] += 1
                else:
                    conditional_prob_dict[key][key2][the_val] = 1
                # conditional_num_dict[key][nLen][key2] = float(conditional_num_dict[key][nLen][key2])
            num_val = len(count_val.keys())
            for x in count_val.keys():
                if x not in conditional_prob_dict[key][key2].keys():
                    conditional_prob_dict[key][key2][x] = 0
            counter = sum(conditional_prob_dict[key][key2].values()) 
            p = 1/num_val
            for key3 in conditional_prob_dict[key][key2].keys():
                example_vi_y = conditional_prob_dict[key][key2][key3] 
                conditional_prob_dict[key][key2][key3] = (example_vi_y + m_estimate*p)/(counter+ m_estimate)
        
        # conditional_prob_dict[key] = conditional_amount_dict
        # if conditional_amount_dict == save:
        #     print("???")
        # save = conditional_amount_dict

        # label - attribute - value:prob
    return conditional_prob_dict


def train_model(attr_type_flag, whole_dataset ,data_value_list, numberD, trans_label,m_estimate): 

    numeric_columns, max_min_dict = make_maxminarray(attr_type_flag, whole_dataset)  # make max min array for numeric value
    threshold_dict = thresholdContinuous(numeric_columns, max_min_dict, numberD, data_value_list, attr_type_flag) # UPDATE data_value_list 
    conditional_prob_dict = conditional_prob(trans_label, data_value_list,m_estimate)

    return conditional_prob_dict, threshold_dict

def continuous_to_discrete(testing_set_value, attr_type_flag, threshold_dict):

    for nNumberD in range(len(testing_set_value)):
        for key in testing_set_value[nNumberD].keys():
            if attr_type_flag[key] == '1':
                a = float(testing_set_value[nNumberD][key])
                if a > threshold_dict[key][0] and threshold_dict[key][1] > a or a == threshold_dict[key][0]:
                    testing_set_value[nNumberD][key] = '0'
                elif a > threshold_dict[key][1] and threshold_dict[key][2] > a or a == threshold_dict[key][1]:
                    testing_set_value[nNumberD][key] = '1'
                elif a > threshold_dict[key][2] and threshold_dict[key][3] > a or a == threshold_dict[key][2]:
                    testing_set_value[nNumberD][key] = '2'
                elif a > threshold_dict[key][3] and threshold_dict[key][4] > a or a == threshold_dict[key][3]:
                    testing_set_value[nNumberD][key] = '3'              
                elif a > threshold_dict[key][4] and threshold_dict[key][5] > a or a == threshold_dict[key][4] or a == threshold_dict[key][5]:
                    testing_set_value[nNumberD][key] = '4'
    return None

def test_model(testing_set_value , testing_set_label ,conditional_prob_dict):
    # conditional_prob_dict : { label: {'attr': { value_i : prob }} }
    # testing_set_label : [ {'attr':'value' }, {'attr':'value' }, ... ]
    example_label_prob = {}
    prediction = {}
    store_prob = 1
    label_num = len(conditional_prob_dict)

    for nExample in range(len(testing_set_value)):
    # testing_set_value[nLen] is dict: {'attr key':'value'}
        example_label_prob[nExample] = {}
        for nLabel in range(label_num):
        # conditional_prob_dict[nLabel] is {'attr': {'value': prob} }
            if nLabel not in conditional_prob_dict.keys():
                store_prob = 0
            else:
                # if nLen not in example_label_prob.keys():
                for key in testing_set_value[nExample].keys():
                    # 'key' means attributes
                    val = testing_set_value[nExample][key]
                    if val not in conditional_prob_dict[nLabel][key].keys():
                        prob = 0
                    else:
                        prob = conditional_prob_dict[nLabel][key][val]
                    store_prob = prob * store_prob
            example_label_prob[nExample][nLabel] = store_prob 
            store_prob = 1

    for nLen in range(len(testing_set_value)):
        # key, val = max(example_label_prob[nLen].items(), key=lambda x: x)
        # prediction[nLen] = key
        # pred, value = max(example_label_prob[nLen],key=lambda key:example_label_prob[nLen][key])
        inverse = [(value, key) for key, value in example_label_prob[nLen].items()]
        prediction[nLen] = max(inverse)[1]

    return prediction

# def a(o,abc):
#     abc.append(o)

#     return abc

# def b():
#     o = 10
#     abc = [1,1,1]
#     xyz = a(o,abc)
#     print(xyz)
#     print(abc)
#     return None

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



        

