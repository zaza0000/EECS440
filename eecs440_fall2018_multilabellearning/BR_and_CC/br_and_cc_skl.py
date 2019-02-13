from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from cross_validation import construct_cv_folds
from skmultilearn.dataset import load_from_arff
from nbayes import pre_process_continuous
from Problem_Transformation import BinaryRelevance, ClassifierChains
import numpy as np
from Metrics import *


def main(**kwargs):
    """
    main function for problem transformation algorithms
    :param kwargs: includes path_to_data,feature_type, num_labels, algorithm
    :return:
    """
    path = kwargs.pop('path_to_data')
    feature_type = kwargs.pop('feature_type')
    num_labels = kwargs.pop('num_labels')
    algorithm = kwargs.pop('algorithm')

    X, y = load_from_arff(path, label_count=num_labels, label_location="end", load_sparse=not feature_type)

    X = X.toarray()
    y = y.toarray()

    if feature_type == 1:
        for i in range(len(X[0])):
            X[:, i] = pre_process_continuous(X[:, i], 5)

    features_list, target_list = construct_cv_folds(5, X, y)
    accuracy = []
    ham_score = []
    precision = []
    recall = []
    f1_score = []

    for i in range(5):
        training_set_features = []
        training_set_target = []
        for j in range(5):
            if i != j:
                training_set_features = training_set_features + features_list[j]
                training_set_target = training_set_target + target_list[j]

        X_train = np.array(training_set_features)
        y_train = np.array(training_set_target)
        X_test = np.array(features_list[i])
        y_test = np.array(target_list[i])

        # Training and Testing
        if feature_type == 0:
            if algorithm == "BR":
                br = BinaryRelevance(BernoulliNB)
                br.fit(X_train, y_train)
                predictions = br.predict(X_test)

            else:
                cc = ClassifierChains(BernoulliNB)
                cc.fit(X_train, y_train)
                predictions = cc.predict(X_test)

        else:
            if algorithm == "BR":
                br = BinaryRelevance(MultinomialNB)
                br.fit(X_train, y_train)
                predictions = br.predict(X_test)

            else:
                cc = ClassifierChains(MultinomialNB)
                cc.fit(X_train, y_train)
                predictions = cc.predict(X_test)
        # print(predictions)
        # print(y_test)

        acc = accuracy_score(y_test, predictions)
        ham = hamming_score(y_test, predictions)
        p, r, f1, _ = precision_recall_fscore_support(y_test, predictions, average='micro')
        accuracy.append(acc)
        ham_score.append(ham)
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    print("Accuracy: " + str(sum(accuracy) / 5))
    print("Hamming_Score: " + str(sum(ham_score) / 5))
    print("Precision: " + str(sum(precision) / 5))
    print("Recall: " + str(sum(recall) / 5))
    print("F1_Score: " + str(sum(f1_score) / 5))


if __name__ == "__main__":
    param1 = {'path_to_data': './Datasets/yeast/yeast.arff', 'feature_type': 1, 'num_labels': 14, 'algorithm': 'BR'}
    param2 = {'path_to_data': './Datasets/yeast/yeast.arff', 'feature_type': 1, 'num_labels': 14, 'algorithm': 'CC'}
    param3 = {'path_to_data': './Datasets/scene/scene.arff', 'feature_type': 1, 'num_labels': 6, 'algorithm': 'BR'}
    param4 = {'path_to_data': './Datasets/scene/scene.arff', 'feature_type': 1, 'num_labels': 6, 'algorithm': 'CC'}
    param5 = {'path_to_data': './Datasets/medical/medical.arff', 'feature_type': 0, 'num_labels': 45, 'algorithm': 'BR'}
    param6 = {'path_to_data': './Datasets/medical/medical.arff', 'feature_type': 0, 'num_labels': 45, 'algorithm': 'CC'}
    param7 = {'path_to_data': './Datasets/enron/enron.arff', 'feature_type': 0, 'num_labels': 53, 'algorithm': 'BR'}
    param8 = {'path_to_data': './Datasets/enron/enron.arff', 'feature_type': 0, 'num_labels': 53, 'algorithm': 'CC'}
    main(**param1)

