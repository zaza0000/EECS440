"""
Metrics for Multi-label Classification problem

"""
# Module Name: nbayes.py
# Author: Zheng Wang
import numpy as np


def hamming_score(y_test, predictions):
    """
    Hamming score is defined as 1 - hamming_loss
    :param y_test:
    :param predictions:
    :return: hamming_score
    """
    return 1.0 - np.mean(predictions != y_test)


def accuracy_score(y_test, predictions):
    total = len(predictions)
    correct = 0
    for i in range(total):
        if np.array_equal(y_test[i], predictions[i]):
            correct += 1
    return correct/total


