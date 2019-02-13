from mldata import *
# from test_mldata import *
from math import log2
import numpy as np
from Naive_Bayes_boosting import *
from scipy.io import arff
import pandas as pd

def main():
    data_read = parse_c45("voting")
    # print(np.shape(data_read))
    #initialized the parameter
    k = 3
    m = 5
    # print(len(data_read))
    weight = 1/ len(data_read) * np.ones(len(data_read))
    weight = weight.reshape(-1,1)
    # print(weight)
    error_ind = []



    save_prediction_label = np.zeros([len(data_read),1])
    save_target_label_list = [x[-1] for x in data_read]
    save_target_label = np.array(save_target_label_list)
    save_target_label = save_target_label.reshape(-1,1)
    # print(save_prediction_label)
    # print(save_target_label)
    # whole_prediction_length = range(len(data_read))
    # print(whole_prediction_length)
    # print(data_read[1])
    # print(data_read)
    # '''
    # test    
    # '''
    # labels_1_ind = np.where(save_target_label == 1)
    # labels_0_ind = np.where(save_target_label == 0)
    # labels_1_ratio = len(labels_1_ind)/(len(labels_1_ind)+len(labels_0_ind))
    # labels_0_ratio = len(labels_0_ind)/(len(labels_1_ind)+len(labels_0_ind))
    # print("target episilon = ", labels_1_ratio,labels_0_ratio)
    # '''
    # test end    
    # '''    
    iteration = 20

    for iiter in range(iteration):    
        for j in range(len(data_read)):
            the_example = data_read[j]
            data_newarray = np.array(data_read.to_float())
            new_data = data_newarray[:, 1:-1]
            maxminarray = np.zeros([2,len(new_data[0])])
            for i in range(1, len(data_read[0])-1):
                if data_read.schema.features[i].type == 'CONTINUOUS':
                    maxminarray[0, i-1] = np.min(new_data[:,i-1])
                    maxminarray[1, i-1] = np.max(new_data[:,i-1])

            labels_ratio, save_all_prob, save_all_threshold = showme_dataset(data_read,k,m,maxminarray,weight,error_ind)

            operated_example = operate_example(data_read,the_example,save_all_threshold)
            predict_label = showme_example(operated_example,save_all_prob, labels_ratio)
            # print(predict_label)
            save_prediction_label[j] = predict_label
            # print(predict_label,j)
        
        # print(save_prediction_label)
        # print(save_target_label)
        # print(save_prediction_label-save_target_label)
        error_ind, a = np.where(save_prediction_label - save_target_label)
        weight,episilon = boosting_updata_weight(weight, error_ind ,save_prediction_label, save_target_label )
        # print(episilon)
        # if episilon[0] == 0.5:
        #     print("EQUAL!!")
        # if episilon[0] > 0.5:
        #     print("GREATER!!")
        if episilon[0] > 0.5 or episilon[0] == 0 or episilon[0] == 0.5:
            print("episilon greater than 0.5, or episilon equal to 0")
            print(save_prediction_label)
            return 
    # print(episilon)
    # print("the true value is :", len(error_ind)/(  ) )
    print(save_prediction_label)

    # path = './yeast/yeast-train.arff'
    # data, meta = arff.loadarff(path)
    # df = pd.DataFrame(data)
    # print(df)
    # print(meta)



if __name__ == '__main__':
    main()