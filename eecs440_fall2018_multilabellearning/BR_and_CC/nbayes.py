"""Main file for the implementation of the Naive Bayes algorithm.

Functions Include:

Classes Include: None.

"""
# Module Name: nbayes.py
# Author: Zheng Wang

import numpy as np


def pre_process_nominal(old_col, values):
    """
    map nominal values to numerical values 1~k
    :param old_col: column vector to be changed
    :param values: all possible nominal value
    :return: new_col: the new column which will later replace old_col
    """
    values_list = list(values)
    new_col = [values_list.index(e) + 1 for e in old_col]

    return new_col


def pre_process_continuous(old_col, k):
    """
    partition the range of the feature into k bins.
    Then replace the feature with a discrete feature that takes value x if the original feature’s value was in bin x.
    :param old_col: column vector to be changed
    :param k: number of bins
    :return: new_col: the new column which will later replace old_col
    """

    max_value = np.max(old_col)         # start point
    min_value = np.min(old_col)         # end point
    # print(min_value, max_value)
    bin_splits = np.linspace(min_value, max_value, num=k+1)
    new_col = []
    for value in old_col:
        for i in range(len(bin_splits)-1):
            if bin_splits[i] <= value < bin_splits[i+1]:
                new_value = i
                new_col.append(new_value)
                break
            elif value == max_value:
                new_value = k-1
                new_col.append(new_value)
                break
    return new_col


def pre_process_class(old_col):
    """
    map class (True/False) to 1/0
    :param old_col: column vector to be changed
    :return: new_col: the new column which will later replace old_col
    """
    new_col = [0 if e == "False" else 1 for e in old_col]
    return new_col


def nbayes(features, target, feature_type, num_bins):
    """
    Naive Bayes main function
    m-estimates are used to estimate P(X=x|Y=y)
    :param features: all features
    :param target: target, (0,1)
    :param feature_type: 0 for nominal, 1 for continuous
    :param num_bins: number of bins for continuous values
    :return: true_dict, false_dict, pr_true, pr_false
                true_dict: is a dictionary storing Pr(X=xi|Y=True) for all X
                false_dict: is a dictionary storing Pr(X=xi|Y=False) for all X
                pr_true: is Pr(Y=True)
                pr_false: is Pr(Y=False)
    """
    """
    Since all features are now transformed to nominal(binary) values, we can calculate all Pr(X=xi|Y=y) easily.
    Well, it seems under m-estimates the procedure can never be easy. Need to take nominal values as special cases!
    """
    # p: is the prior estimate of the probability
    true_dict = {}
    false_dict = {}
    unique, counts = np.unique(target, return_counts=True)
    # First calculate P(Y=y)
    if len(unique) == 2:
        pr_true = counts[1] / (counts[1] + counts[0])
        pr_false = counts[0] / (counts[1] + counts[0])
    else:
        if 0 in unique:
            pr_true = 0
            pr_false = 1
        else:
            pr_false = 0
            pr_true = 1

    # Now calculate Pr(X=xi|Y=y) for all X
    pos_examples = features[np.where(target == 1)]
    neg_examples = features[np.where(target == 0)]
    num_features = np.size(features, axis=1)  # number of features
    total_pos = len(pos_examples)  # used to calculate conditional probabilities
    total_neg = len(neg_examples)
    if feature_type == 0:
        possible_values = (0, 1)
    else:
        possible_values = [i[0] for i in enumerate(range(num_bins), start=0)]
    for i in range(num_features):  # real index of feature is i+1
        true_dict[i] = {}
        col = list(pos_examples[:, i])
        counts = []
        for value in possible_values:
            counts.append(col.count(value))
        v = len(possible_values)
        p = 1 / v

        m = v  # use Laplace smoothing, m = v

        for j in range(v):  # {X: {x1: Pr(X=x1|Y=True), x2: Pr(X=x2|Y=True)}}
            true_dict[i][possible_values[j]] = (counts[j] + m * p) / (total_pos + m)
            # true_dict[i][possible_values[j]] = counts[j] / total_pos

        false_dict[i] = {}
        col = list(neg_examples[:, i])
        counts = []
        for value in possible_values:
            counts.append(col.count(value))
        v = len(possible_values)
        p = 1 / v

        m = v  # use Laplace smoothing, m = v
        for j in range(v):  # {X: {x1: Pr(X=x1|Y=True), x2: Pr(X=x2|Y=True)}}
            false_dict[i][possible_values[j]] = (counts[j] + m * p) / (total_neg + m)
            # false_dict[i][possible_values[j]] = counts[j] / total_neg

    return true_dict, false_dict, pr_true, pr_false


def nbayes_prediction(nbayes_param, features):
    true_dict, false_dict, pr_true, pr_false = nbayes_param
    pr_x_given_true = 0  # use log probability to avoid floating point underflow
    pr_x_given_false = 0
    prediction = []
    for feature in features:
        for j in range(len(feature)):
            pr_x_given_true += np.log(true_dict[j][feature[j]])
            pr_x_given_false += np.log(false_dict[j][feature[j]])
        predict_true = pr_x_given_true + np.log(pr_true)
        predict_false = pr_x_given_false + np.log(pr_false)
        predict = 1 if predict_true >= predict_false else 0
        prediction.append(predict)
    prediction = np.array(prediction)
    return prediction
