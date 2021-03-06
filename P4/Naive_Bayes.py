from mldata import parse_c45
from mldata import ExampleSet
from math import log2
import numpy as np
import math


'''
function: 
    This function is used to discretize the continuous values into K bins, and calculate the p(x=xi| y=+), p(x=xi| y=-)
input: 
    1. dataset:  the whole dataset, include the ID (first column) and label (last column).
    2. maxminarray:  the max and min of each attribute
    3. k:  K bins
    4. whichAttribute:  choose an attribute
output: 
    1.features: 
        shape:  examples number *  2 
        type: array
        indexs: [value(1 to k), labels]
    2.values_ratio:
        shape:  values number * 4
        type: array
        indexs: [value, p(x = xi), p(x=xi| y=+), p(x=xi| y=-)]
'''

def continuous_values(attribute_with_labels, k, m, maxminarray):

    attribute_with_labels = attribute_with_labels[attribute_with_labels[:,0].argsort()]
    rows, columns = np.shape(attribute_with_labels) # columns number is 2 more than attributes
    max = maxminarray[1]
    min = maxminarray[0]
    ranges = max - min
    kDevideRange = ranges/k
    threshold = min + kDevideRange
    i = 1
    count = 0
    features = attribute_with_labels
    save_threshold = []
    for j in range(rows):
        if (features[j,0] < threshold) or (features[j,0] == threshold):
            features[j,0] = i         
        else:
            save_threshold.append(threshold)
            threshold += kDevideRange
            i += 1        
            features[j,0] = i     
    labels = attribute_with_labels[:,1] 
    # the array contains values and labels


    values, ratio = np.unique(features[:,0], return_counts = True)
    values_ratio_temp = np.array([values, ratio/rows])
    # this array contain values and percent.
    values_ratio = np.zeros([len(values), 4])
    values_ratio[:,:-2] = np.transpose(values_ratio_temp)

    # if np.size(np.unique(labels)) == 1:
    #     return None,[[None,0,0],[None,0,0]]

    labels_1_ind = np.where(labels == 1)
    labels_0_ind = np.where(labels == 0)
    labels_1_ratio = len(labels_1_ind)/(len(labels_1_ind)+len(labels_0_ind))
    labels_0_ratio = len(labels_0_ind)/(len(labels_1_ind)+len(labels_0_ind))
    # print(labels_1_ratio,labels_0_ratio)

    value_poslabel = features[labels_1_ind[0],:]
    values_poslabel, number_poslabel = np.unique(np.array(value_poslabel)[:,0],return_counts = True)
    value_neglabel = features[labels_0_ind[0],:]
    values_neglabel, number_neglabel = np.unique(np.array(value_neglabel)[:,0],return_counts = True)

    if (m > 0) or (m == 0):
        j = 0
        for i in range(len(values)):
            if(j < len(number_poslabel)):
                if values_poslabel[j] == values[i]:
                    values_ratio[i,2] = (number_poslabel[j]+m/len(values)) / (len(labels_1_ind[0]) + m)
                    j += 1
                else:
                    values_ratio[i,2] = (m/len(values)) / (len(labels_1_ind[0]) + m)
            else:
                values_ratio[i,2] = (m/len(values)) / (len(labels_1_ind[0]) + m)

        j = 0
        for i in range(len(values)):
            if(j < len(number_neglabel)):
                if values_neglabel[j] == values[i]:
                    values_ratio[i,3] = (number_neglabel[j]+m/len(values)) / (len(labels_0_ind[0]) + m)
                    j += 1
                else:
                    values_ratio[i,3] = (m/len(values)) / (len(labels_0_ind[0]) + m)
            else:          
                values_ratio[i,3] = (m/len(values)) / (len(labels_0_ind[0]) + m)
    else:
        j = 0
        for i in range(len(values)):
            if(j < len(number_poslabel)):
                if values_poslabel[j] == values[i]: 
                    values_ratio[i,2] = (number_poslabel[j]+ 1 ) / (len(labels_1_ind[0]) +len(values))
                    j += 1
                else:
                    values_ratio[i,2] = 1 / (len(labels_1_ind[0]) + len(values))
            else:
                values_ratio[i,2] = 1 / (len(labels_1_ind[0]) + len(values))

        j = 0
        for i in range(len(values)):
            if(j < len(number_neglabel)):
                if values_neglabel[j] == values[i]:
                    values_ratio[i,3] = (number_neglabel[j]+1) / (len(labels_0_ind[0]) + len(values))
                    j += 1
                else:
                    values_ratio[i,3] = 1 / (len(labels_0_ind[0]) + len(values))
            else:          
                values_ratio[i,3] = 1 / (len(labels_0_ind[0]) + len(values))


    return [labels_1_ratio,labels_0_ratio], values_ratio, save_threshold


'''
function: 
    This function is used to calculate the p(x=xi| y=+), p(x=xi| y=-) for discrete values.
'''
def discrete_values(attribute_with_labels, m):
    attribute_with_labels = attribute_with_labels[attribute_with_labels[:,0].argsort()]
    rows, columns = np.shape(attribute_with_labels) # columns number is 2 more than attributes
    values, count = np.unique(attribute_with_labels[:,0], return_counts = True)
    values_ratio_temp = np.array([values, count/rows])
    values_ratio = np.zeros([len(values), 4])
    values_ratio[:,:-2] = np.transpose(values_ratio_temp)

    labels = attribute_with_labels[:,1]
    labels_1_ind = np.where(labels == 1)
    labels_0_ind = np.where(labels == 0)
    labels_1_ratio = len(labels_1_ind[0])/ (len(labels_1_ind[0])+len(labels_0_ind[0]))
    labels_0_ratio = len(labels_0_ind[0])/ (len(labels_1_ind[0])+len(labels_0_ind[0]))
 
    value_poslabel = attribute_with_labels[labels_1_ind[0],:]
    values_poslabel, number_poslabel = np.unique(np.array(value_poslabel)[:,0],return_counts = True)
    value_neglabel = attribute_with_labels[labels_0_ind[0],:]
    values_neglabel, number_neglabel = np.unique(np.array(value_neglabel)[:,0],return_counts = True)

    j = 0
    for i in range(len(values)):
        if(j < len(number_poslabel)):
            if values_poslabel[j] == values[i]:
                values_ratio[i,2] = (number_poslabel[j]+m/len(values)) / (len(labels_1_ind[0]) + m)
                j += 1
            else:
                values_ratio[i,2] = (m/len(values)) / (len(labels_1_ind[0]) + m)
        else:
            values_ratio[i,2] = (m/len(values)) / (len(labels_1_ind[0]) + m)

    j = 0
    for i in range(len(values)):
        if(j < len(number_neglabel)):
            if values_neglabel[j] == values[i]:
                values_ratio[i,3] = (number_neglabel[j]+m/len(values)) / (len(labels_0_ind[0]) + m)
                j += 1
            else:
                values_ratio[i,3] = (m/len(values)) / (len(labels_0_ind[0]) + m)
        else:
            values_ratio[i,3] = (m/len(values)) / (len(labels_0_ind[0]) + m)

    return [labels_1_ratio,labels_0_ratio], values_ratio




'''
Output: 
    save_all_prob:
        The saved prob list is like below
        [[ [p+,p-], array([ values, precent, P(v|y=+), P(v|y=-)]) ],
        [ [p+,p-], array([ values, precent, P(v|y=+), P(v|y=-)]) ] ]
        Each row presents different attribute

Function: This function is used to operate the full dateset. When receive dataset, this function will         calculate P(x=xi|y=+) and P(x=xi|y=-), and then use a list to save them.
'''
def showme_dataset(data_read, k, m, maxminarray, new_chosen_trainSet, chosen_columns, nUnchosenColumns, save_continuous):

    data_array = np.array(data_read.to_float())
    data_array = np.delete(data_array,0,1)

    labels = data_array[:,-1]
    labels_1_ind = np.where(labels == 1)
    labels_0_ind = np.where(labels == 0)
    labels_1_ratio = len(labels_1_ind[0])/ (len(labels_1_ind[0])+len(labels_0_ind[0]))
    labels_0_ratio = len(labels_0_ind[0])/ (len(labels_1_ind[0])+len(labels_0_ind[0]))
    label_ratio = [labels_1_ratio,labels_0_ratio]

    none_row, none_column = np.where(data_array == None)
    rows, columns = np.shape(data_array)
    save_array = data_array
    save_all_prob = []
    save_all_threshold = []

    for i in range(len(chosen_columns)):
        data_array = save_array  
        if new_chosen_trainSet.ndim == 1:
            attribute_with_labels = np.transpose(np.array([new_chosen_trainSet[i],new_chosen_trainSet[-1]]))
        else:
            attribute_with_labels = np.transpose(np.array([new_chosen_trainSet[:,i],new_chosen_trainSet[:,-1]])) 
        if chosen_columns[i] in save_continuous:
            labels_ratio, values_ratio, save_threshold = continuous_values(attribute_with_labels, k,m,maxminarray[:,chosen_columns[i]])
            save_all_threshold = np.append(save_all_threshold, save_threshold, axis = 0 )
        else: 
            labels_ratio, values_ratio = discrete_values(attribute_with_labels, m)
 
        save_all_prob.append([values_ratio])

    return  label_ratio ,save_all_prob, save_all_threshold


def showme_example(the_example,save_all_prob, label_ratio, chosen_columns):
    
    # the_example = np.array(the_example.to_float())
    # the_example = np.delete(the_example,0)
    if (None in the_example) == True :
        return None

    final_array = label_ratio
    # for i in range(len(chosen_columns)):
    #     values_ratio = (save_all_prob[i])[0]

    #     for j in range(len(values_ratio)):
    #         if ( the_example[i] == values_ratio[j,0] ):
    #             final_array = np.vstack((final_array, np.array([values_ratio[j,2],values_ratio[j,3]])) )
    for nChosen_columns in range(len(chosen_columns)):
        values_ratio = (save_all_prob[nChosen_columns])[0]
        ratios = np.hstack((  np.transpose(np.array([values_ratio[:,2]])), np.transpose(np.array([values_ratio[:,3]]))   ))
        final_array = np.vstack((  final_array, ratios  ))
    
    predict_label = naive_calculate(final_array)

    return predict_label



'''
input: values_ratio
    shape: value number * 4
    type: array
    index: value, P(X= xi), P(X=xi|y=1), P(X=xi|y=0)
output: predict_label
function: calculate the possibility of postive label and negtive label, and compare the possibility.
'''
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

def operate_example(data_read,the_example,save_all_threshold,chosen_columns, save_continuous):
    
    count = 0
    flag = 0
    try:
        columns, _ = np.size(save_all_threshold, axis = 0)
        flag = 1
    except TypeError:
        columns = np.size(save_all_threshold, axis = 0)

    for i in range(len(chosen_columns)):   
        if chosen_columns[i] in save_continuous:
            count += 1
            for j in range(columns):
                if flag == 0:
                    if the_example[i] < save_all_threshold[j]:
                        the_example[i] = j+1
                        break
                else:
                    if the_example[i] < save_all_threshold[count-1, j]:
                        the_example[i] = j+1
                        break

    return the_example


# def choose_new_dataset(data_array, ki, chosen_columns, columns):
#     if chosen_columns == -1:
#     else: 
#         for 
#     return newDataArray, updateChosenColumns

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
    return validationSet, trainSet

def makeUnchosenColumns(columns, chosen_columns):
    whole_column = list(range(1,columns))
    if chosen_columns == None:
        return whole_column
    for i in chosen_columns:
        whole_column.remove(i)
    return whole_column

# import list and output array
def chosen_dataset(validationSet, trainSet, chosen_columns):
    chosen_trainSet = np.transpose(np.array([trainSet[:,-1]]))
    chosen_validationSet = np.transpose(np.array([validationSet[:,-1]]))
    if chosen_columns != None:
        for nChosen_columns in chosen_columns:
            # print(np.shape(np.transpose(np.array([trainSet[:,nChosen_columns]]))))
            # print(np.shape(chosen_trainSet))
            chosen_trainSet = np.hstack(( np.transpose(np.array([trainSet[:,nChosen_columns]])),chosen_trainSet ))
            chosen_validationSet = np.hstack(( np.transpose(np.array([validationSet[:, nChosen_columns]])), chosen_validationSet))
    return chosen_trainSet, chosen_validationSet


def splitDataset(data_newarray, rows):
    cv_list = [[],[],[]]
    chunk_size = math.floor(rows/3)
    cv_list[0] = data_newarray[ 0:chunk_size ,:]
    cv_list[1] = data_newarray[ chunk_size:2*chunk_size ,:]
    cv_list[2] = data_newarray[ 2*chunk_size:-1 ,:]
    # print(len(cv_list[0]),len(cv_list[1]),len(cv_list[2]))

    return cv_list
