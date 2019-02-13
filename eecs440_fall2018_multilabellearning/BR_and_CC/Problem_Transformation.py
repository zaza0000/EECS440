import numpy as np
from nbayes import pre_process_continuous
import copy


class BinaryRelevance:
    """
    Used with sklearn classifiers
    """
    def __init__(self, base_learner, params={}):
        self.base_learner = base_learner
        self.params = params
        self.n = 0  # number of labels
        self.classifiers = list()

    def fit(self, x_train, y_train):
        """
        The BR method first create n (the number of labels) classifiers, each is a base learner.
        In training phase, the i-th classifier is trained with all features as attributes, and i-th label as target.
        """
        self.n = y_train.shape[1]
        # generate n of classifiers
        self.classifiers = [self.base_learner(**self.params) for i in range(self.n)]
        for i in range(self.n):  # train these clfs with n labels
            self.classifiers[i].fit(x_train, y_train[:, i])

    def predict(self, x_test):
        """
        the j-th classifier classifies the j-th label
        :param x_test: test features
        :return: predictions
        """
        predictions = np.zeros((x_test.shape[0], self.n), dtype=int)
        for i in range(self.n):
            predictions[:, i] = self.classifiers[i].predict(x_test)
        return predictions


class ClassifierChains:
    """
    Used with sklearn classifiers
    """
    def __init__(self, base_learner, params={}):
        self.base_learner = base_learner
        self.params = params
        self.n = 0
        self.classifiers = []

    def fit(self, x_train, y_train):
        self.n = y_train.shape[1]
        self.classifiers = [self.base_learner(**self.params) for i in range(self.n)]
        for i in range(self.n):
            self.classifiers[i].fit(np.concatenate((x_train, y_train[:, :i]), axis=1), y_train[:, i])

    def predict(self, x_test):
        predictions = np.zeros((x_test.shape[0], self.n), dtype=int)
        for i in range(self.n):
            predictions[:, i] = self.classifiers[i].predict(np.concatenate((x_test, predictions[:, :i]), axis=1))
        return predictions


class BinaryRelevanceNB:
    """
    Used with self implemented classifiers
    """
    def __init__(self, learner, predictor, params={}):
        self.learner = learner
        self.params = params
        self.n = 0  # number of labels
        self.predictor = predictor
        self.p_params = []

    def fit(self, x_train, y_train):
        self.params["features"] = x_train
        self.n = y_train.shape[1]
        # generate n of classifiers and train them
        for i in range(self.n):
            self.params["target"] = y_train[:, i]
            self.p_params.append({"nbayes_param": self.learner(**self.params)})

    def predict(self, x_test):
        predictions = np.zeros((x_test.shape[0], self.n), dtype=int)
        for i in range(self.n):
            p_param = self.p_params[i]
            p_param["features"] = x_test
            predictions[:, i] = self.predictor(**p_param)
        return predictions


class ClassifierChainsNB:
    """
    Used with self implemented classifiers
    """
    def __init__(self, learner, predictor, params={}):
        self.learner = learner
        self.params = params
        self.n = 0  # number of labels
        self.predictor = predictor
        self.p_params = []
        self.feature_type = params['feature_type']

    def fit(self, x_train, y_train):
        self.n = y_train.shape[1]
        # generate n of classifiers and train them
        for i in range(self.n):
            # pre-process y_train[:, :i] if continuous
            if self.feature_type == 1 and i >= 1:
                y_train[:, i-1] = pre_process_continuous(y_train[:, i-1], 2)
            self.params["features"] = np.concatenate((x_train, y_train[:, :i]), axis=1)
            self.params["target"] = y_train[:, i]
            self.p_params.append({"nbayes_param": self.learner(**self.params)})

    def predict(self, x_test):
        predictions = np.zeros((x_test.shape[0], self.n), dtype=int)
        temp = copy.deepcopy(predictions)
        for i in range(self.n):
            if self.feature_type == 1 and i >= 1:
                temp[:, i-1] = pre_process_continuous(predictions[:, i-1], 2)
            p_param = self.p_params[i]
            p_param["features"] = np.concatenate((x_test, temp[:, :i]), axis=1)
            predictions[:, i] = self.predictor(**p_param)
        return predictions



