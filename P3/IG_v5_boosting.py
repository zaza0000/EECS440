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


def discrete_data(data_fil,line_num,weight):
    '''
    This function is used to 
    1. arrange data based on one discrete attribute
    2. calculate the ratio of each value on the attribute
    3. data must be binary or nominal
    Tip: the output are arranged array and [values,ratio] 
    '''
    v_pos = 0
    data_fil = np.c_[data_fil,weight]
    data_arranged = data_fil[data_fil[:,line_num].argsort()]
    rows,columns = np.shape(data_arranged)
    values,values_places,counts = np.unique(data_arranged[:,line_num], return_counts=True, return_inverse = True) # values and their number
    
    temp = 0
    j = 0
    counts_temp = []
    for i in counts:
        counts_temp.append( sum(weight[temp:i+temp]))
        temp = i
        j += 1
    ratio = counts/sum(counts) # ratio of each value

    weight = [x[-1] for x in data_arranged]
    weight = np.array(weight)
    all_label = [x[-2] for x in data_arranged]
    all_label_value, all_label_num, label_ind = np.unique(all_label, return_counts = True, return_inverse = True) # number of all postive label
    label_ind = np.array(all_label)
    weight_with_poslabel = weight[np.where(label_ind ==1)]

    numerator = 0.0
    denominator = 0.0
    rows_values = np.size(values,0)
    ratio_pos = np.zeros(rows_values)
    j = 0
    for i in range(rows):
        # print("rows = ",rows)
        # print(" i =", i)
        # print(data_arranged[i,line_num])

        if i == rows-1 :
            if(label_ind[i] == 1):
                numerator += weight[i]
            denominator +=  weight[i]
            ratio_pos[j] = numerator/denominator
            j +=1
            break  
        if (data_arranged[i,line_num] != data_arranged[i+1,line_num]):
            if(label_ind[i] == 1):
                numerator += weight[i]
            denominator +=  weight[i]             
            ratio_pos[j] = numerator/denominator
            j +=1
        # print(j)
        # print(data_arranged[i,line_num])
    # P (x = v)
    # rows_values = np.size(values,0)
    # values_pos = np.zeros(rows_values)
    # num = 0
    # for i in range(rows):
    #     if i < rows-1: 
    #         if (data_arranged[i,-2] == 1):
    #             num += 1
    #         if data_arranged[i,line_num] != data_arranged[i+1,line_num]:
    
    #             values_pos[values_places[i]] = sum(weight[temp: temp+ num])
    #             temp = num
    #             num = 0
    #     elif i == rows-1:
    #         if (data_arranged[i,-2] == 1):
    #             num += 1
    #         if data_arranged[i,line_num] == data_arranged[i-1,line_num]:
    #             values_pos[values_places[i]] += num
    # ratio_pos = np.true_divide (values_pos, counts)

    temp = np.array([values,ratio,ratio_pos])
    values_ratio = temp.T  
    # print(values_ratio)
    return data_arranged, values_ratio


def continuous_data(data_fil,line_num,weight):
    '''
    This function is used to
    1. arrange data based on one continuous attribute
    2. find out where the label is changed and calculate the average of values between there
    3. data must be contiunous
    Tip: the output are arranged array and [values,ratio]
    '''
    data_fil = np.c_[data_fil,weight]
    data_arranged = data_fil[data_fil[:,line_num].argsort()]
    rows = np.size(data_arranged,0)
    values_ratio=[]
    count = 0
    weight = [x[-1] for x in data_arranged]
    weight = np.array(weight)
    all_label = [x[-2] for x in data_arranged]
    all_label_value, all_label_num, label_ind = np.unique(all_label, return_counts = True, return_inverse = True) # number of all postive label
    # print( all_label_value, all_label_num, label_ind )
    label_ind = np.array(all_label)
    weight_with_poslabel = weight[np.where(label_ind ==1)]

    numerator = 0.0
    denominator = 0.0
    pos_sum_weight = sum(weight_with_poslabel)
    if np.size(np.unique(all_label)) == 1:
        return data_arranged,[[None,0,0],[None,0,0]]
    else:
        for i in range(rows-1):
            if (data_arranged[i,-2] != data_arranged[i+1,-2]):
                if(label_ind[i] == 1):
                    numerator += weight[i]
                denominator +=  weight[i]             
                # if sum(weight[0:i-1]) != 0 and sum(weight[i:-1]) != 0:
                values_ratio.append([  data_arranged[i,line_num]/2+data_arranged[i+1,line_num]/2, denominator, numerator /denominator ])
                values_ratio.append([data_arranged[i,line_num]/2+data_arranged[i+1,line_num]/2, 1-denominator, (pos_sum_weight-numerator)/(1-denominator)])
            # else:
            #     values_ratio.append( [data_arranged[i,line_num]/2+data_arranged[i-1,line_num]/2, 0 , 0 ])
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
    # print(hyx)
    return hyx

def find_minima_threshold(value_ratio):
    '''
    This function will be used only when the attribute is CONTINUOUS,
    and can find the threshold which have minima H(Y|X)
    '''
    value_per_pos = []
    save_hyx = []
    rows = np.shape(value_ratio)[0]
    for i in range(rows-1):
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
        


def information_gain(data_fil, hyx, weight):
    '''
    This funcion is used to calculate information gain.
    '''
    label_column = [x[-1] for x in data_fil]
    label_value, label_number = np.unique(label_column,return_inverse = True )
    if label_number.size == 0:
        return 0
    label_column = np.array(label_column)
    position = np.where(label_column == 1)
    # print(position)
    ratio_pos = sum(weight[position]) / sum(weight)
    # print(np.where(label_column == 1))
    # print(ratio_pos)
    # print(sum(weight[position]))
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
        hxv = [x*0 for x in p]
        #print(hxv)
        #print(p)
        for i in range(len(p)):
            if p[i] != 0:
                hxv[i] = p[i] * np.log2(p[i])
            else:
                hxv[i] = 0
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

