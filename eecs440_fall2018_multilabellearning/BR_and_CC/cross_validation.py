import random
import numpy as np


def construct_cv_folds(n, data_value_list, data_label_list):
    datasets_value = []
    datasets_label = []
    for i in range(n):
        datasets_value.append([])
        datasets_label.append([])
    num_data = len(data_value_list)
    data_label_list_copy = []
    for i in range(num_data):
        data_label_list_copy.append(data_label_list[i].copy())
    flag = []
    stop_flag = []
    for i in range(num_data):
        flag.append(0)
        stop_flag.append(1)
    while flag != stop_flag:
        for i in range(num_data):
            if flag[i] == 0:
                flag[i] = 1
                same_labels = [i]
                example_label = data_label_list_copy[i]
                break
        for i in range(1, len(data_label_list_copy)):
            if flag[i] == 0 and np.array_equal(data_label_list_copy[i], example_label):
                same_labels.append(i)
                flag[i] = 1
        one_piece = len(same_labels) / n
        shuffle(same_labels)
        for i in range(n):
            dataset_value = []
            dataset_label = []
            for j in range(int(i * one_piece), int((i + 1) * one_piece)):
                dataset_value.append(data_value_list[same_labels[j]])
                dataset_label.append(data_label_list[same_labels[j]])
            shuffle(dataset_value)
            shuffle(dataset_label)
            datasets_value[i] = datasets_value[i] + dataset_value
            datasets_label[i] = datasets_label[i] + dataset_label
    return datasets_value, datasets_label


def shuffle(dataset):
    random.seed(12345)
    random.shuffle(dataset)
