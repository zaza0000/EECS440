import random
import copy
import build_tree_BFS
from mldata import *
from constants import *
from logreg_BFS import Logistic_Regression
from operator import itemgetter
import numpy as np
import time

err_rates = []
ROC_matrix = []
time_consumed = []

def load_data():
    path_name = DATA_PATH.rpartition('/')
    path = path_name[0]
    name = path_name[2]
    full_dataset = parse_c45(name, path)
    return ExampleSet(full_dataset)

def forward_festure_selection(datasets = None, algorithm = 1, testing_set = None):
	if(datasets == None or testing_set == None):
		return None
	if(algorithm == 1):    #1=dtree, 2=nbayes, 3=logreg
		print("Algorithm: Decision Tree")
		prediction, true_label = dtree(datasets, testing_set)
	elif(algorithm == 2):
		print("Algorithm: Naive Bayes")
		prediction = nbayes(datasets, testing_set)
	elif(algorithm == 3):
		print("Algorithm: Logistic Regression")
		prediction, true_label = logreg(datasets, testing_set)
	else:
		print("wrong algorithm index")
		return None

	return prediction, true_label

# ====================================
# dtree():
#   decision tree
# ====================================
def dtree(training_set, testing_set):
	# TODO: call functions from the first programming
	selected_features = list()
	feature_set = dict()
	is_improved = 1
	max_feature_num = len(training_set[0])-2
	for i in range(max_feature_num):
		feature_set[i+1] = 0

	pre_error_rate = -1

	time_start = time.time()
	while(len(selected_features) != min(NUM_OF_FEATURES, max_feature_num) and is_improved == 1):
		feature_set = three_fold_val_dTree(feature_set, training_set, selected_features)
		#print(feature_set)
		attriIndex = min(feature_set,key=feature_set.get) 
		new_error_rate = feature_set.get(attriIndex)
		if(pre_error_rate != -1):
			if(pre_error_rate >= new_error_rate):
				#print("previous error rate: ", pre_error_rate, "after delete a feature(the best): ", new_error_rate)
				pre_error_rate = new_error_rate
				selected_features.append(attriIndex)
				feature_set.pop(attriIndex)
				for i in feature_set:
					feature_set[i] = 0
			else:
				is_improved = 0
		else:
			pre_error_rate = new_error_rate
			selected_features.append(attriIndex)
			feature_set.pop(attriIndex)
			for i in feature_set:
				feature_set[i] = 0
	time_end = time.time()
	time_ = time_end - time_start
	time_consumed.append(time_)
	print("Time consumed:",time_, 's')

	prediction = list()
	tree = build_tree_BFS.build_DecisionTree(MAX_DEPTH, EPS, training_set, ENABLE_GAIN, attri_indices = selected_features)
	size = tree.get_tree_size()
	max_depth = tree.get_tree_depth()
	first_feature_index = tree.get_root().get_attriIndex()
	first_feature = training_set.schema.features[first_feature_index].name
	#prediction = list(map(tree.classify_data, testing_set))
	print('Size: %d ,Maximum Depth: %d ,First Feature: %s' % (size, max_depth, first_feature))	

	print("selected_features: ", end=' ')
	for key in selected_features:
		print(key, end=' ')
	print()

	prediction = list(map(tree.classify_data, testing_set))
	true_label = np.array(testing_set.to_float())[:,-1]
	return prediction, true_label

def three_fold_val_dTree(feature_set, training_data, selected_features):
	for key in feature_set:
		temp = copy.deepcopy(selected_features)
		temp.append(key)
		new_data_set = fold_n_cv(training_data, 3)
		for i in range(3):
			real_train = new_data_set[i]
			for index in range(len(new_data_set[(i+1)%3])):
				real_train.append(new_data_set[(i+1)%3][index])
			real_val = new_data_set[(i+2)%3]

			tree = build_tree_BFS.build_DecisionTree(MAX_DEPTH, EPS, real_train, ENABLE_GAIN, attri_indices = temp)

			acc = tree.classify_dataset(real_val)
			#print(acc)

			feature_set[key] = feature_set[key] + acc
		feature_set[key] = feature_set[key]/3
		feature_set[key] = 1 - feature_set[key]
	return feature_set

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

def three_fold_val_LR(feature_set, training_data, real_train_set, schema):
	for key in feature_set:
		temp = np.c_[real_train_set, training_data[:,key]]
		temp = np.c_[temp, training_data[:,-1]]
		new_data_set = np.array_split(temp,3)
		for i in range(3):
			real_train = np.concatenate([new_data_set[(i)%3], new_data_set[(i+1)%3]], axis=0)
			real_val = new_data_set[(i+2)%3]
			lg = Logistic_Regression(lambdaa = LAMBDA, training_data = real_train, schema = schema, iteration = ITER2, learning_rate = LR)
			pre, true_label = lg.classify_data(real_val, schema)
			acc = 0
			for i in range(len(pre)):
				if(pre[i] >= 0.5 and true_label[i] == 1):
					acc = acc + 1
				elif(pre[i] < 0.5 and true_label[i] == 0):
					acc = acc + 1

			feature_set[key] = feature_set[key] + acc/len(pre)
		feature_set[key] = feature_set[key]/3
		feature_set[key] = 1 - feature_set[key]
	return feature_set


def updata_dataset(attriIndex, training_data, real_train, testing_set, real_test):
	real_train = np.c_[real_train, training_data[:,attriIndex]]
	real_test = np.c_[real_test, testing_set[:,attriIndex]]
	feature_set = dict()
	max_feature_num = len(training_data[0])-2
	for i in range(max_feature_num):
		feature_set[i+1] = 0
	return feature_set, training_data, real_train, testing_set, real_test

# ====================================
# logreg():
#   logistic regression
# ====================================
def logreg(dataset, testing_set):
	# TODO: call functions from the second programming
	selected_features = list()
	feature_set = dict()
	is_improved = 1
	max_feature_num = len(dataset[0])-2
	for i in range(max_feature_num):
		feature_set[i+1] = 0

	pre_error_rate = -1
	data = dataset
	data2 = testing_set
	training_data = np.array(data.to_float())
	real_train = np.array(training_data[:,0])
	testing_data = np.array(data2.to_float())
	real_test = np.array(testing_data[:,0])
	
	time_start = time.time()
	while(len(selected_features) != min(NUM_OF_FEATURES, max_feature_num) and is_improved == 1):
		feature_set = three_fold_val_LR(feature_set, training_data, real_train, data.schema)
		attriIndex = min(feature_set,key=feature_set.get) 
		new_error_rate = feature_set.get(attriIndex)
		if(pre_error_rate != -1):
			if(pre_error_rate >= new_error_rate):
				#print("previous error rate: ", pre_error_rate, "after delete a feature(the best): ", new_error_rate)
				pre_error_rate = new_error_rate
				feature_set, training_data, real_train, testing_data, real_test = updata_dataset(attriIndex, training_data, real_train, testing_data, real_test)
				selected_features.append(attriIndex)
			else:
				is_improved = 0
		else:
			pre_error_rate = new_error_rate
			feature_set, training_data, real_train, testing_data, real_test = updata_dataset(attriIndex, training_data, real_train, testing_data, real_test)
			selected_features.append(attriIndex)
	time_end = time.time()
	time_ = time_end - time_start
	time_consumed.append(time_)
	print("Time consumed:", time_, "s")
	real_train = np.c_[real_train, training_data[:, -1]]
	real_test = np.c_[real_test, testing_data[:, -1]]
	
	prediction = list()
	lg = Logistic_Regression(lambdaa = LAMBDA, training_data = real_train, schema = data.schema, iteration = ITER2, learning_rate = LR)
	pre, true_label = lg.classify_data(real_test, data.schema)
	prediction = pre
	print("selected_features: ", end=' ')
	for key in selected_features:
		print(key, end=' ')
	print()
	return prediction, true_label

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

def main():
	#Error value processing
	if(ALGORITHM != 1 and ALGORITHM != 2 and ALGORITHM != 3):
		raise ValueError("ALGORITHM must be 1 or 2 or 3")
	if(NUM_OF_FEATURES <= 0):
		raise ValueError("NUM_OF_FEATURES must be positive")
	elif(type(NUM_OF_FEATURES) != int):
		raise ValueError("NUM_OF_FEATURES must be an integer")

	#load data
	full_dataset = load_data()

	datasets = fold_n_cv(full_dataset, 5)
	accuracies, precisions, recalls = cv(datasets)
	avg_accuracy = 0
	avg_precision = 0
	avg_recall = 0
	std_accuracy = 0
	std_precision = 0
	std_recall = 0
	avg_time_consumed = 0
	for i in range(5):
		avg_accuracy += accuracies[i]
		avg_precision += precisions[i]
		avg_recall += recalls[i]
		avg_time_consumed += time_consumed[i]
	avg_accuracy = avg_accuracy / 5
	avg_precision = avg_precision / 5
	avg_recall = avg_recall / 5
	avg_time_consumed = avg_time_consumed / 5
	for i in range(5):
		std_accuracy += (accuracies[i] - avg_accuracy) ** 2
		std_precision += (precisions[i] - avg_precision) ** 2
		std_recall += (recalls[i] - avg_recall) ** 2
	std_accuracy = (std_accuracy / 5) ** 0.5
	std_precision = (std_precision / 5) ** 0.5
	std_recall = (std_recall / 5) ** 0.5
	ROC_area = compute_ROC_area()
			
	print("Avg Time Consumed: %.3f s"%avg_time_consumed)
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
		val_data = datasets[i]
		shuffle(train_data)
		shuffle(val_data)
		probs, true_label = forward_festure_selection(datasets = train_data, algorithm = ALGORITHM, testing_set = val_data)
		accuracy, precision, recall = get_results(probs, true_label)
		accuracies.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
		print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
		
	return accuracies, precisions, recalls

def compute_ROC_area():
	ROC_value = []
	ROC_matrix.sort(key = itemgetter(0), reverse = True)
	for threshold in ROC_matrix:
		true_positive = 0
		true_negative = 0
		false_negative = 0
		false_positive = 0
		confidence_level = threshold[0]
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
		
def fold_n_cv(full_dataset, n):		#Divide full_dataset into stratified 5-folds
	#Separate the full_dataset into two sets in terms of the label
	true_set = ExampleSet(ex for ex in full_dataset if ex[-1] == True)
	false_set = ExampleSet(ex for ex in full_dataset if ex[-1] == False)
	shuffle(true_set)
	shuffle(false_set)
	
	#Calculate the length of each set
	true_len = len(true_set)
	true_len_part = true_len / n
	false_len = len(false_set)
	false_len_part = false_len / n
	
	datasets = []
	
	for i in range(n):
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


if __name__ == '__main__':
    main()