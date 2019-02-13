from mldata import parse_c45
# from test_mldata import ExampleSet
from math import log2
import numpy as np

'''
Version 5.3 functions:
1. add find_minima_threshold function, to find out which threshold have largest IG
2. revise continuous_data function, make the return values easier to read.
3. add threshold of largest information gain
4. gain ratio function
5. make the output easier to read.
'''

def filter_for_None(data_array):
    '''
    This function is used to 
    delete all of the 'None' in array
    '''
    i,j = np.where(data_array == None)
    data_fil = np.delete(data_array,i,0)
    return data_fil


def discrete_data(data_fil,line_num):
    '''
    This function is used to 
    1. arrange data based on one discrete attribute
    2. calculate the ratio of each value on the attribute
    3. data must be binary or nominal
    Tip: the output are arranged array and [values,ratio] 
    '''
    v_pos = 0
    data_arranged = data_fil[data_fil[:,line_num].argsort()]
    rows,columns = np.shape(data_arranged)
    values,values_places,counts = np.unique(data_arranged[:,line_num], return_counts=True, return_inverse = True) # values and their number
    ratio = counts/sum(counts) # ratio of each value
    rows_values = np.size(values,0)
    values_pos = np.zeros(rows_values)

    num = 0
    for i in range(rows):
        if i < rows-1: 
            if (data_arranged[i,-1] == 1):
                num += 1
            if data_arranged[i,line_num] != data_arranged[i+1,line_num]:

                values_pos[values_places[i]] = num
                num = 0
        elif i == rows-1:
            if (data_arranged[i,-1] == 1):
                num += 1
            if data_arranged[i,line_num] == data_arranged[i-1,line_num]:
                values_pos[values_places[i]] += num
    ratio_pos = np.true_divide (values_pos, counts)

    temp = np.array([values,ratio,ratio_pos])
    values_ratio = temp.T  

    return data_arranged, values_ratio


def continuous_data(data_fil,line_num):
    '''
    This function is used to
    1. arrange data based on one continuous attribute
    2. find out where the label is changed and calculate the average of values between there
    3. data must be contiunous
    Tip: the output are arranged array and [values,ratio]
    '''
    data_arranged = data_fil[data_fil[:,line_num].argsort()]
    rows = np.size(data_arranged,0)
    values_ratio=[]
    count = 0
    all_label = [x[-1] for x in data_arranged]
    all_label_value, all_label_num = np.unique(all_label, return_counts = True) # number of all postive label
    if np.size(np.unique(all_label)) == 1:
        return data_arranged,[[None,0,0],[None,0,0]]
    else:
        for i in range(rows):
            if (data_arranged[i-1,-1] != data_arranged[i,-1]) and (i != 0 and i !=  [rows]):
                count = 0
                for j in range(i):
                    if all_label[j] == 1:
                        count += 1 
                values_ratio.append([data_arranged[i,line_num]/2+data_arranged[i-1,line_num]/2, i/rows, count/i])
                values_ratio.append([data_arranged[i,line_num]/2+data_arranged[i-1,line_num]/2, (1-i/rows),(all_label_num[1]-count)/(rows-i)] )
        # value_ratio = ([threshold, percent, pos_ratio])       
        return data_arranged, values_ratio

def calculate_hyx(values_ratio):
    '''
    This function is used to calculate H(Y|X)
    '''
    ratio_pos = [x[2] for x in values_ratio]
    entropy_pos = []
    entropy_neg = []
    for x in ratio_pos:
        if (x != 0) and (x != 1) :
            temp_pos = x * np.log2(x)
            temp_neg = (1-x) * np.log2(1-x)
            entropy_pos.append(temp_pos)
            entropy_neg.append(temp_neg)
        if (x == 0) or (x == 1):
            entropy_pos.append(0)
            entropy_neg.append(0)
    hyx_v = np.add(entropy_neg, entropy_pos)
    hyxv = [x*(-1) for x in hyx_v]# H(Y|X=v)
    ratio_attribute = [x[1] for x in values_ratio]
    hyx = sum(np.multiply(ratio_attribute,hyxv))

    return hyx

def find_minima_threshold(value_ratio):
    '''
    This function will be used only when the attribute is CONTINUOUS,
    and can find the threshold which have minima H(Y|X)
    '''
    value_per_pos = []
    save_hyx = []
    rows = np.shape(value_ratio)[0]
    for i in range(rows):
        if i %2 == 0:
            value_per_pos.append(value_ratio[i]) 
            value_per_pos.append(value_ratio[i+1])
            save_hyx.append(calculate_hyx(value_per_pos))
            value_per_pos = []
    # print('the H(Y|X) = ',save_hyx)
    wanted_hyx = min(save_hyx)
    where_min_hyx = np.argmin(save_hyx)
    threshold = value_ratio[where_min_hyx*2][0]

    return threshold, wanted_hyx
        


def information_gain(data_fil, hyx):
    '''
    This funcion is used to calculate information gain.
    '''
    label_column = [x[-1] for x in data_fil]
    label_value, label_number = np.unique(label_column,return_counts = True)
    if label_number.size == 0:
        return 0
        
    ratio_pos = label_number[0]/sum(label_number)
    ratio_neg = 1 - ratio_pos
    # print(ratio_pos,ratio_neg)
    if (ratio_pos == 0) or (ratio_neg == 0):
        return -hyx
    hy = - (ratio_pos)* np.log2(ratio_pos) - ratio_neg* np.log2(ratio_neg)
    ig = hy - hyx
    # print('H(Y) = ',hy)

    return ig


def gain_ratio(ig, value_ratio):
    '''
    This function is used to calculate gain ratio .
    '''
    p = [x[1] for x in value_ratio ]
    if len(p) == 1:
        return 0
    else:
        hxv = p * np.log2(p)
        if p[0]+p[1] != 1:        
            hx = -sum(hxv)
        else:
            hx = -sum(hxv[::2])
        # print('H(X)=', hx)
    
    if hx == 0 or hx == 1:
        g_ratio = 0
    else:
        g_ratio = ig / hx   

    return g_ratio

