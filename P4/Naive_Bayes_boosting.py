from mldata import parse_c45
from mldata import ExampleSet
from math import log2
import numpy as np


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

def continuous_values(attribute_with_labels, k, m, maxandmin, weight, attribute_index):
    # print(maxandmin)
    attribute_with_labels = np.c_[attribute_with_labels,weight]
    attribute_with_labels = attribute_with_labels[attribute_with_labels[:,0].argsort()]
    rows, columns = np.shape(attribute_with_labels) # columns number is 2 more than attributes
    max_con = maxandmin[1]
    min_con = maxandmin[0]
    ranges = max_con - min_con
    kDevideRange = ranges/k
    threshold = min_con + kDevideRange

    i = 1
    count = 0
    features = attribute_with_labels
    save_threshold, features = build_threshold(threshold, features, kDevideRange, rows, k)

    values = np.unique(attribute_with_labels[:,0])
    save_values = []
    values_ratio =[]
    for vvalues in values:
        save_values = build_subset(attribute_with_labels, attribute_index , vvalues)
        value_ratio = compute_labelpercent(save_values, m, 1/len(values))
        values_ratio.append(value_ratio)    
    
    return  values_ratio, save_threshold


def build_threshold(threshold, features, kDevideRange, rows, k):
    save_threshold = [threshold]
    i = 1
    for j in range(rows):
        if (features[j,0] > threshold) and (i < k):
            threshold += kDevideRange
            save_threshold.append(threshold)
            i += 1       
        features[j,0] = i     
    return save_threshold, features


def build_subset(dataset, attribute_index, attribute_value):
    subset = np.zeros([1,len(dataset[0])])
    for i in range(len(dataset)):
        if(dataset[i][0] == attribute_value):
            subset = np.vstack((subset, dataset[i]))
    subset = np.delete(subset,0,0)

    return subset

def compute_labelpercent(save_values, m, p):
    if len(save_values) == 0:
        return 0
    pospercent_numerator = 0.0
    pospercent_denominator = 0.0
    values_ratio = []
    array_temp = save_values[0]
    for k in range(len(save_values)):
        if save_values[k,1] == 1:
            pospercent_numerator += save_values[k,2]
        pospercent_denominator += save_values[k,2]
 
    values_ratio.append( [array_temp[0], (pospercent_denominator+m*p)/(1+m), pospercent_numerator/pospercent_denominator, (pospercent_denominator-pospercent_numerator)/pospercent_denominator ])
    # print(values_ratio[0])
    return values_ratio[0]

'''
function: 
    This function is used to calculate the p(x=xi| y=+), p(x=xi| y=-) for discrete values.
'''
def discrete_values(attribute_with_labels, m, weight, attribute_index):

    attribute_with_labels = np.c_[attribute_with_labels,weight]
    values = np.unique(attribute_with_labels[:,0])
    save_values = []
    values_ratio =[]
    for vvalues in values:
        save_values = build_subset(attribute_with_labels, attribute_index , vvalues)
        value_ratio = compute_labelpercent(save_values, m, 1/len(values))
        values_ratio.append(value_ratio)    
    return values_ratio

'''
Output: 
    save_all_prob:
        The saved prob list is like below
        [[ [p+,p-], array([ values, precent, P(v|y=+), P(v|y=-)]) ],
        [ [p+,p-], array([ values, precent, P(v|y=+), P(v|y=-)]) ] ]
        Each row presents different attribute

Function: This function is used to operate the full dateset. When receive dataset, this function will         calculate P(x=xi|y=+) and P(x=xi|y=-), and then use a list to save them.
'''
def showme_dataset(data_read,k,m,maxminarray,weight,error_ind):

    data_array = np.array(data_read.to_float())
    data_array = np.delete(data_array,0,1)

    labels = data_array[:,-1]
    # example(Y = 1 or 0)
    labels_1_ind = np.where(labels == 1)
    labels_0_ind = np.where(labels == 0)
    # percent :  P( Y = 1 or 0)
    labels_1_values = sum(weight[np.where(labels ==1)])
    labels_0_values = sum(weight[np.where(labels ==0)])
    labels_1_ratio = labels_1_values/(labels_1_values+labels_0_values)
    labels_0_ratio = labels_0_values/(labels_1_values+labels_0_values)
    label_ratio = [[labels_1_ratio[0],labels_0_ratio[0]]]
    # print(label_ratio)
    none_row, none_column = np.where(data_array == None)
    rows, columns = np.shape(data_array)
    save_array = data_array
    save_all_prob = []
    save_all_threshold = [[]]
    features = 0

    for i in (range(columns-1)):
        data_array = save_array  
        for x in none_column:
            if x == i:
                index = np.where(none_column == i)
                data_array = np.delete(data_array,none_row[index],0)
        attribute_with_labels = np.transpose(np.array([data_array[:,i],data_array[:,-1]]))
        # print(attribute_with_labels)    
        if data_read.schema[i+1].type == 'CONTINUOUS':
            values_ratio, save_threshold = continuous_values(attribute_with_labels, k,m,maxminarray[:,i], weight, i)
            save_all_threshold.append(save_threshold)
        else:
            values_ratio = discrete_values(attribute_with_labels, m , weight, i)
        # print("values_ratio = ",values_ratio,"\n")
        save_all_prob.append(values_ratio)  
    # print(save_all_prob)
    return  label_ratio ,save_all_prob, save_all_threshold


def showme_example(the_example, save_all_prob, label_ratio):
    the_example = np.array(the_example.to_float())
    the_example = np.delete(the_example,0)
    final_array = label_ratio.copy()
    if (None in the_example) == True :
        return None
    for i in range(len(the_example)-1):
        values_ratio = save_all_prob[i]
        for j in range(len(values_ratio)):
            if ( the_example[i] == (values_ratio[j])[0] ):
                # print(np.array([ (values_ratio[j])[2],(values_ratio[j])[3] ]))
                final_array.append( [ (values_ratio[j])[2],(values_ratio[j])[3] ] ) 
    # print(final_array)
    predict_label = naive_calculate(final_array)
    # print(predict_label)
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
    p_x_1 = np.prod([x[0] for x in final_array])
    p_x_0 = np.prod([x[1] for x in final_array])
    if (p_x_1 > p_x_0) :
        return 1
    elif (p_x_1 < p_x_0):
        return 0
    else: 
        return  np.random.randint(2,size=1)

def operate_example(data_read,the_example,save_all_threshold):
    
    count = 0
    flag = 0
    try:
        columns, _ = np.size(save_all_threshold, axis = 0)
        flag = 1
    except TypeError:
        columns = np.size(save_all_threshold, axis = 0)

    for i in range( len(the_example)-1 ):   
        if data_read.schema[i].type == 'CONTINUOUS':
            count += 1
            threshold_for_example = save_all_threshold[count]
            # print(threshold_for_example)
            for nlen in range(len(threshold_for_example)):
                if float(the_example[i]) < threshold_for_example[nlen]:
                    the_example[i] = nlen+1
                    break
                if the_example[i] == threshold_for_example[-1] or the_example[i] > threshold_for_example[-1]:
                    the_example[i] = len(threshold_for_example)
                    break
    # print(the_example)
    return the_example

def boosting_updata_weight( weight, error_ind, save_prediction_label, save_target_label ):
    episilon = 0
    count = 0
    error_ind = np.where(save_prediction_label != save_target_label)
    #print(len(error_ind[0]))
    save_prediction_label = np.where(save_prediction_label == 0, -1, 1)
    save_target_label = np.where(save_target_label == 0, -1, 1)

    y_hx = save_prediction_label * save_target_label
    rows_0,col0  = np.where(y_hx == -1)

    for i in rows_0 :
        episilon = episilon + weight[i]  
    if episilon == 0 or episilon ==1:
        alpha = 0
    else:
        alpha = 0.5 * np.log10( (1-episilon) /episilon )
    
    y_multiply_hx = save_prediction_label * save_target_label
    # print(y_multiply_hx)
    weight_temp = weight * np.exp(-alpha * y_multiply_hx)
    # print(weight_temp)
    weight_normalized = weight_temp/(sum(weight_temp))
    # print(weight_normalized)

    return weight_normalized, episilon, alpha
            

