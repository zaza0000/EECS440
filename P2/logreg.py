import time
from mldata import *
from constants import *
from nbayes import *
import mldata
import numpy as np
from math import log

err_rates = []

# ====================================
# load_data() 
# description:
#   load data from files
# ====================================
def load_data():
    path_name = DATA_PATH.rpartition('/')
    path = path_name[0]
    name = path_name[2]
    full_dataset = mldata.parse_c45(name, path)
    return ExampleSet(full_dataset)

def shuffle(datalist):          #shuffle datalist with random seed 12345
    random.seed(12345)
    random.shuffle(datalist)
# ====================================
# normalize_and_encode(original_data):
# description:
#   control the range of the data
# ====================================
def normalize_and_encode(original_data):
    data = np.array(original_data.to_float())
    stds = data.std(axis = 0)
    means = data.mean(axis = 0)
    for data_index in range(0, len(data)):
        for i in range(1, data[data_index].size - 1):
            if original_data.schema.features[i].type == 'CONTINUOUS':
                data[data_index][i] = (data[data_index][i] - means[i]) / stds[i]
    normalized_data = data[ :, 1 : -1]
    label = data[ :, -1 : ]
    return normalized_data, label, stds, means
# ====================================
# sigmoid(x):
# description:
#   Sigmoid function
# ====================================
def sigmoid(x):
    if(x>=0):
        return 1.0/(1 + np.exp(-x))
    else:
        return np.exp(x)/(1 + np.exp(x))

# ====================================
# likelihood(example, label, weights):
# description:
#   use sigmoid to compute likelihood
# ====================================
def likelihood(example, label, weights, bias):       
    likeli = sigmoid(np.dot(weights, example)+bias)
    if(label > 0):
        return likeli
    else:
        return 1 - likeli
# ====================================
# nitialize_weights(examples):
# description:
#   use random to get initial weights
# ====================================
def initialize_weights(examples):
    random = np.random.RandomState()
    return random.rand(examples.shape[1])
# ==================================================
# def optimize_function(examples, labels, initial_weights):
# description:
#   use gradient descent to minimize the loss_function
# ==================================================
def optimize_function(examples, labels, initial_weights, bias, lambdaa, iteration, learning_rate):
    new_weights = initial_weights;
    new_bias = bias;
    #print(loss_function(initial_weights, bias, lambdaa, examples, labels))
    while(iteration > 0):
        iteration -= 1
        db = derivative_bias(initial_weights, bias, examples, labels)
        bias = bias - learning_rate*db
        for i in range(0, len(initial_weights)):
            dw = derivative_weight(initial_weights, i, bias, lambdaa, examples, labels)
            initial_weights[i] = initial_weights[i] - learning_rate*dw

    #print(loss_function(initial_weights, bias, lambdaa, examples, labels))

    return new_weights, new_bias

def derivative_bias(weights, bias, dataset, label):
    db = 0
    for data_index in range(0, len(dataset)):
        sig = sigmoid(np.dot(weights, dataset[data_index ])+bias)
        db += sig - label[data_index]

    return db

def derivative_weight(weights, weight_index, bias, lambdaa, dataset, label):
    temp = 0
    for data_index in range(0, len(dataset)):
        sig = sigmoid(np.dot(weights, dataset[data_index])+bias)
        temp += (sig - label[data_index])*dataset[data_index][weight_index]
    dw = temp + lambdaa*weights[weight_index]

    return dw
# ====================================
# loss_function(weights, bias, lambdaa, dataset):
# description:
#   loss_function: the negative conditional log 
#   likelihood plus a constant (λ) times a penalty term, 
#   half of the 2-norm of the weights squared.
# ====================================
def loss_function(weights, bias, lambdaa, dataset, label):
    likelihood_sum = 0
    penalty_term = 0
    for data_index in range(0, len(dataset)):
        likeli = likelihood(dataset[data_index], label[data_index], weights, bias)
        if(likeli != 0):
            likelihood_sum+= np.log(likeli)
        else:
            likelihood_sum += 0
    for weight_index in range(0, len(weights)):
        penalty_term += weights[weight_index]*weights[weight_index]
    penalty_term = 0.5*lambdaa*penalty_term

    return -likelihood_sum + penalty_term
# ====================================
# compute_accuracy(num_TP, num_FP, num_TN, num_FN):
# description:
#   compute accuracy
# ====================================
def compute_accuracy(num_TP, num_FP, num_TN, num_FN):
    if(num_TP + num_TN == 0):
        return 0
    accuracy = (num_TP + num_TN) / (num_TP + num_FP + num_TN + num_FN)
    return accuracy
# ====================================
# compute_precision(num_TP, num_FP):
# description:
#   compute precision
# ====================================
def compute_precision(num_TP, num_FP):
    if(num_TP == 0):
        return 0
    precision = num_TP / (num_TP + num_FP)
    return precision
# ====================================
# compute_recall(num_TP, num_FN)
# description:
#   compute recall
# ====================================
def compute_recall(num_TP, num_FN):
    if(num_TP == 0):
        return 0
    recall = num_TP / (num_TP + num_FN)
    return recall
# ====================================
# contingency_table(prediction, true_label):
# description:
#   get a contingency table
# ====================================
def contingency_table(prediction, true_label):
    tp, fn, fp, tn = 0, 0, 0, 0
    for pred, label in zip(prediction, true_label):
        if label == 1 and pred > 0.5:
            tp += 1
        elif label == 1 and pred <= 0.5:
            fn += 1
        elif label == 0 and pred > 0.5:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn

# ====================================
# class Logistic_Regression(object):
# description:
#   Logistic_Regression
# ====================================
class Logistic_Regression(object):

    def __init__(self, lambdaa=1, training_data = None, iteration = 10, learning_rate = 0.05):
        self.lambdaa = lambdaa
        self.means = 0
        self.stds = 0
        self.weights = None
        self.iteration = iteration
        if(len(training_data) == 0):
            print("No data input")
            return

        normalized_data, data_label, stds, means = normalize_and_encode(training_data)
        self.means = means
        self.stds = stds
        self.weights = initialize_weights(normalized_data)
        np.random.seed() 
        self.bias = np.random.uniform(-10,10);
        self.learning_rate = learning_rate;
        #print("initial_weights:", self.weights)
        self.train_weights(normalized_data, data_label)



    def train_weights(self, normalized_data, data_label):
        #print("step2: training...")
        self.weights, self.bias = optimize_function(normalized_data, data_label, self.weights, self.bias, self.lambdaa, self.iteration, self.learning_rate)
        #optimize_function(examples, labels, initial_weights)

    def classify_data(self, examples):
        #print("step3: classify")
        data = np.array(examples.to_float())
        for data_index in range(0, len(data)):
            for i in range(1, data[data_index].size - 1):
                if examples.schema.features[i].type == 'CONTINUOUS':
                    data[data_index][i] = (data[data_index][i] - self.means[i]) / self.stds[i]
        normalized_data = data[ :, 1 : -1]
        label = data[ :, -1 : ]
        prediction = list(map(self.classify_data2, normalized_data))
        return prediction, label



    def classify_data2(self, example):
        prob = sigmoid(np.dot(self.weights, example))
        return prob


        
# ====================================
# get_results(results):
# description:
#   
# ====================================
def get_results(prediction, true_label):
    global ROC_matrix
    for i in range(len(prediction)):
        ROC_matrix.append([prediction[i], true_label[i]])
    global err_rates
    num_TP, num_FP, num_TN, num_FN = 0, 0, 0, 0
    num_TP, num_FN, num_FP, num_TN = contingency_table(prediction, true_label)
    acc = compute_accuracy(num_TP, num_FP, num_TN, num_FN)
    #print ("accuracy: ", acc)
    prec = compute_precision(num_TP, num_FP)
    #print ("precision: ", prec)
    rec = compute_recall(num_TP, num_FN)
    #print ("recall: ", rec)
    err_rate = 1 - acc
    err_rates.append(err_rate)
    return acc, prec, rec


def cross_logreg(original_data):
    datasets = fold_5_cv(original_data)
    accuracies = []
    precisions = []
    recalls = []
    for i in range(5):
        train_data = mldata.ExampleSet()
        for j in range(1, 5):
            for index in range(len(datasets[(i + j) % 5])):
                train_data.append(datasets[(i + j) % 5][index])
        val_data = datasets[i]
        shuffle(train_data)
        shuffle(val_data)
        lg = Logistic_Regression(lambdaa = LAMBDA, training_data = train_data, iteration = ITER, learning_rate = LR)
        predictions, true_label = lg.classify_data(val_data)
        accuracy, precision, recall = get_results(predictions, true_label)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
    return accuracies, precisions, recalls


def main():

    #print ("step1: load data")
    original_data = load_data()  #load data

    if(ENABLE_VAL == 1):

        lg = Logistic_Regression(lambdaa = LAMBDA, training_data = original_data, iteration = ITER, learning_rate = LR)
        predictions, true_label = lg.classify_data(original_data)
        accuracy, precision, recall = get_results(predictions, true_label)
        ROC_area = compute_ROC_area()
        print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
    elif(ENABLE_VAL == 0):
        accuracies, precisions, recalls = cross_logreg(original_data)
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        std_accuracy = 0
        std_precision = 0
        std_recall = 0
        for i in range(5):
            avg_accuracy += accuracies[i]
            avg_precision += precisions[i]
            avg_recall += recalls[i]
        avg_accuracy = avg_accuracy / 5
        avg_precision = avg_precision / 5
        avg_recall = avg_recall / 5
        for i in range(5):
            std_accuracy += (accuracies[i] - avg_accuracy) ** 2
            std_precision += (precisions[i] - avg_precision) ** 2
            std_recall += (recalls[i] - avg_recall) ** 2
        std_accuracy = (std_accuracy / 5) ** 0.5
        std_precision = (std_precision / 5) ** 0.5
        std_recall = (std_recall / 5) ** 0.5
        ROC_area = compute_ROC_area()
            
        print("Accuracy: %.3f %.3f\nPrecision: %.3f %.3f\nRecall: %.3f %.3f\nArea under ROC: %.3f\n" % (avg_accuracy, std_accuracy, avg_precision, std_precision, avg_recall, std_recall, ROC_area))
        
    #print (training_data)
    #print (data_set)
    #normalized_data, data_label, _, _ = normalize_and_encode(original_data)
    #print(normalized_data)

    #lg = Logistic_Regression(lambdaa = 0.1, training_data = original_data, iteration = 100, learning_rate = 0.02)
    #predictions, true_label = lg.classify_data(original_data)
    #get_ROC(predictions, true_label)

    #print(nitialize_weights(normalized_data))
    #print(normalized_data)


    #print (data_set)
if __name__ == '__main__':
    main()
	
def compute_err_rates():
    if(ENABLE_VAL == 1):
        raise ValueError("ENABLE_VAL should be 0")
    print("Logistic Regression:")
    main()
    return err_rates
