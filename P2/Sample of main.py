from mldata import *
from test_mldata import *
from math import log2
import numpy as np
from Naive_Bayes import *


def main():
    data_read = parse_c45("volcanoes")
    k = 3
    m = 5
    the_example = data_read[50]

    data_newarray = np.array(data_read.to_float())
    new_data = data_newarray[:, 1:-1]
    maxminarray = np.zeros([2,len(new_data[0])])
    for i in range(1, len(data_read[0])-1):
        if data_read.schema.features[i].type == 'CONTINUOUS':
            maxminarray[0, i-1] = np.min(new_data[:,i-1])
            maxminarray[1, i-1] = np.max(new_data[:,i-1])

    labels_ratio, save_all_prob, save_all_threshold = showme_dataset(data_read,k,m,maxminarray)
    operated_example = operate_example(data_read,the_example,save_all_threshold)
    predict_label = showme_example(operated_example,save_all_prob, labels_ratio)
    print(predict_label)




if __name__ == '__main__':
    main()