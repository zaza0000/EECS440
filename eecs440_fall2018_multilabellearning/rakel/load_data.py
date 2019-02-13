from scipy.io import arff
import pandas as pd
import random
import numpy as np
import math

def random_select_labels(data_label_list, k ):
    all_keys = []
    for key in data_label_list[1].keys():
        all_keys.append(int(key))
    random_k_labels =random.sample(all_keys,k)
    random_k_labels_str = [str(x) for x in random_k_labels]
    return random_k_labels_str 

def reverse_to_one_hot(label_array,k,numberD):
    trans_label = {}
    for nNumberD in range(numberD):
        trans_label[str(nNumberD)] = 0
        for ki in range(k):
            trans_label[str(nNumberD)] += math.pow(2, k-1-ki) * label_array[nNumberD][ki]
    # print(trans_label)
    return trans_label
            

def label_powerset(data_label_list, k, random_k_labels_str, flag ):
    numberD = len(data_label_list)
    M = len(data_label_list[1])
    label_array = np.zeros([numberD,k])
    if flag == 1:
        random_k_labels_str = random_select_labels(data_label_list, k )
    for i in range(numberD):
        for j in range(len(random_k_labels_str)):
            label_array[i][j] = data_label_list[i][random_k_labels_str[j]]
    trans_label = reverse_to_one_hot(label_array,k,numberD)   
    # transfered label is array: array(numberD,1) 
    
    return trans_label, random_k_labels_str, M, numberD

def convert_to_binary(prediciton, k):

    binary_prediction = {}
    for key in prediciton.keys():
        binary_prediction[key] = {}
        remain = prediciton[key]
        for ki in range(k):
           # k-1-ki
            if remain > math.pow(2, k-1-ki) or remain == math.pow(2, k-1-ki):
                binary_prediction[key][str(ki)] = 1
                remain -= math.pow(2, k-1-ki)
            else:
                binary_prediction[key][str(ki)] = 0

    return binary_prediction
