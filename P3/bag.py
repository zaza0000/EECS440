import time
import numpy as np
from mldata import *
#from nbayes import *
from constants import *
import build_tree
from logreg import Logistic_Regression
import random
import Naive_Bayes_boosting
from operator import itemgetter

err_rates = []
ROC_matrix = []

# ====================================
# load_data():
#   load data from files
# ====================================
def load_data():
    path_name = DATA_PATH.rpartition('/')
    path = path_name[0]
    name = path_name[2]
    full_dataset = parse_c45(name, path)
    return ExampleSet(full_dataset)

# ====================================
# randPick():
#   random select data from the original data,
#   build a new set with the same size
# ====================================
def randPick(original_data):
	random = np.random.RandomState()
	subset = ExampleSet()
	for i in range(len(original_data)):
		subset.append(original_data[random.randint(0,len(original_data))])
	return subset

# ====================================
# create_k_trainingsets():
#   create several number of traning datasets
# ====================================
def create_k_trainingsets(original_data=None, classifier_number = NUM_BAG):
	if(original_data == None):
		return
	k_trainingsets = dict();
	for i in range(classifier_number):
		k_trainingsets[i] = randPick(original_data)

	return k_trainingsets

# ====================================
# dtree():
#   decision tree
# ====================================
def dtree(training_set, testing_set):
	# TODO: call functions from the first programming
	prediction = list()
	tree = build_tree.build_DecisionTree(MAX_DEPTH, EPS, training_set, ENABLE_GAIN)
	size = tree.get_tree_size()
	max_depth = tree.get_tree_depth()
	first_feature_index = tree.get_root().get_attriIndex()
	first_feature = training_set.schema.features[first_feature_index].name
	#prediction = list(map(tree.classify_data, testing_set))
	#print('Accuracy: %.4f ,Size: %d ,Maximum Depth: %d ,First Feature: %s' % (acc, size, max_depth, first_feature))	
	prediction = list(map(tree.classify_data, testing_set))
	return prediction

# ====================================
# nbayes():
#   naive bayes
# ====================================
def nbayes(training_set, testing_set):
	# TODO: call functions from the second programming
	prediction = list()

	#Calculate the min and man values of each attribute, in order to decide the boundaries of k-bins 
	min_and_max = []
	np_full_dataset = np.array(training_set)
	attr_length = len(training_set.schema) - 2
	min_and_max = np.zeros((attr_length, 2))
	for i in range(1, attr_length + 1):
		if(training_set.schema[i].type == "CONTINUOUS"):
			row = np_full_dataset[:,i].astype(float)
			max = np.amax(row)
			min = np.amin(row)
			min_and_max[i-1][0]=min
			min_and_max[i-1][1]=max
	min_and_max = np.transpose(min_and_max)

	weight=np.ones([len(training_set),1])
	label_ratio, save_all_prob, save_all_threshold = Naive_Bayes_boosting.showme_dataset(training_set, NUM_BINS, M, min_and_max, weight,None)
	for data in testing_set:
		pred = Naive_Bayes_boosting.showme_example(data, save_all_prob, label_ratio)
		if(pred >= 0.5):
			prediction.append(1)
		else:
			prediction.append(0)
	return prediction

# ====================================
# logreg():
#   logistic regression
# ====================================
def logreg(training_set, testing_set):
	# TODO: call functions from the second programming
	prediction = list()
	lg = Logistic_Regression(lambdaa = LAMBDA, training_data = training_set, iteration = ITER2, learning_rate = LR)
	pre, true_label = lg.classify_data(testing_set)
	prediction = pre
	return prediction

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

# ====================================
# bagging:
#   using the bagging algorithm to do the classification
# ====================================
def bagging(datasets = None, algorithm = 1, classifier_number = 1, testing_set = None):
	if(datasets == None or testing_set == None):
		return None
	if(algorithm == 1):    #1=dtree, 2=nbayes, 3=logreg
		print("Algorithm: Decision Tree")
	elif(algorithm == 2):
		print("Algorithm: Naive Bayes")
	elif(algorithm == 3):
		print("Algorithm: Logistic Regression")
	else:
		print("wrong algorithm index")
		return None

	prediction_vote = list()
	prediction_vote = [0] * len(testing_set)
	for i in range(classifier_number):
		prediction = list()
		if(algorithm == 1):
			prediction = dtree(datasets[i], testing_set)
		elif(algorithm == 2):
			prediction = nbayes(datasets[i], testing_set)
		elif(algorithm == 3):
			prediction = logreg(datasets[i], testing_set)
		prediction_vote = list(map(lambda x :x[0]+x[1] ,zip(prediction_vote, prediction)))

	voting_result = list()
	for i in range(len(prediction_vote)):
		voting_result.append(prediction_vote[i]/classifier_number)
	true_label = [x[-1] for x in testing_set]
	
	#accuracy, precision, recall = get_results(voting_result, true_label)
	#ROC_area = compute_ROC_area()
	#print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
	return voting_result, true_label
	
#Choose confidence level 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.
def compute_ROC_area():
	ROC_value = []
	for i in range(0,11):
		true_positive = 0
		true_negative = 0
		false_negative = 0
		false_positive = 0
		confidence_level = 0.1 * i
		for pair in ROC_matrix:
			if(pair[0] >= confidence_level):
				if(pair[1] == 1):
					true_positive += 1
				else:
					false_positive += 1
			else:
				if(pair[1] == 1):
					false_negative += 1
				else:
					true_negative += 1
		TP_rate = 0
		FP_rate = 0
		if(true_positive != 0):
			TP_rate = true_positive / (true_positive + false_negative)
		if(false_positive != 0):
			FP_rate = false_positive / (true_negative + false_positive)
		ROC_value.append([TP_rate, FP_rate])
	ROC_value = sorted(ROC_value, key = itemgetter(1))
	length = len(ROC_value)
	ROC_area = 0
	for i in range(length - 1):
		ROC_area += (ROC_value[i][0] + ROC_value[i + 1][0]) * (ROC_value[i + 1][1] - ROC_value[i][1]) / 2
	return ROC_area
	

def main():

    #Error value processing
	if(ENABLE_VAL != 0 and ENABLE_VAL != 1):
		raise ValueError("ENABLE_VAL must be 0 or 1")
	if(ALGORITHM != 1 and ALGORITHM != 2 and ALGORITHM != 3):
		raise ValueError("ALGORITHM must be 1 or 2 or 3")
	if(ITER <= 0):
		raise ValueError("ITER must be positive")
	elif(type(ITER) != int):
		raise ValueError("ITER must be an integer")

	#Read data
	full_dataset = load_data()
	
	#Build models
	if(ENABLE_VAL == 1):
		k_datasets = create_k_trainingsets(full_dataset)
		voting_result, true_label = bagging(datasets = k_datasets, algorithm = ALGORITHM, classifier_number = NUM_BAG, testing_set = full_dataset)
		accuracy, precision, recall = get_results(voting_result, true_label)
		ROC_area = compute_ROC_area()
		print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
	elif(ENABLE_VAL == 0):
		datasets = fold_5_cv(full_dataset)
		accuracies, precisions, recalls = cv(datasets)
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

def cv(datasets):
	accuracies = []
	precisions = []
	recalls = []
	for i in range(5):
		train_data = ExampleSet()
		for j in range(1, 5):
			for index in range(len(datasets[(i + j) % 5])):
				train_data.append(datasets[(i + j) % 5][index])
		if(P != 0):
			for data in train_data:
				if(random.random() <= P):
					if(data[-1] == True):
						data[-1] = False
					elif(data[-1] == False):
						data[-1] = True
		val_data = datasets[i]
		shuffle(train_data)
		shuffle(val_data)
		k_datasets = create_k_trainingsets(train_data)
		voting_result, true_label = bagging(datasets = k_datasets, algorithm = ALGORITHM, classifier_number = NUM_BAG, testing_set = val_data)
		accuracy, precision, recall = get_results(voting_result, true_label)
		accuracies.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
		print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
		
	return accuracies, precisions, recalls
		
def fold_5_cv(full_dataset):		#Divide full_dataset into stratified 5-folds
	#Separate the full_dataset into two sets in terms of the label
	true_set = ExampleSet(ex for ex in full_dataset if ex[-1] == True)
	false_set = 	ExampleSet(ex for ex in full_dataset if ex[-1] == False)
	shuffle(true_set)
	shuffle(false_set)
	
	#Calculate the length of each set
	true_len = len(true_set)
	true_len_part = true_len / 5
	false_len = len(false_set)
	false_len_part = false_len / 5
	
	datasets = []
	
	for i in range(5):
		dataset = ExampleSet()
		for j in range(int(i * true_len_part), int((i + 1) * true_len_part)):
			dataset.append(true_set[j])
		for j in range(int(i * false_len_part), int((i + 1) * false_len_part)):
			dataset.append(false_set[j])
		datasets.append(dataset)

	return datasets
	
def shuffle(datalist):			#shuffle datalist with random seed 12345
	random.seed(12345)
	random.shuffle(datalist)

def compute_err_rates():
	if(ENABLE_VAL == 1):
		raise ValueError("ENABLE_VAL should be 0")
	print("Bagging")
	main()
	return err_rates

if __name__ == '__main__':
    main()