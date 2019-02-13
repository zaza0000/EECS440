from constants import *
import mldata
import random
import Naive_Bayes
import numpy as np
from operator import itemgetter

#This is the main file for naive bayes
#To run, directly run this file,
#python nbayes.py

ROC_matrix = []
err_rates = []

def main():
	#Error value processing
	if(ENABLE_VAL != 0 and ENABLE_VAL != 1):
		raise ValueError("ENABLE_VAL should be 0 or 1")
	if(NUM_BINS < 2):
		raise ValueError("NUM_BINS should be greater that 2")
	elif(type(NUM_BINS) != int):
		raise TypeError("NUM_BINS should be an integer")

	#Read data
	path_name = DATA_PATH.rpartition('/')
	path = path_name[0]
	name = path_name[2]
	full_dataset = mldata.parse_c45(name, path)
	
	#Calculate the min and man values of each attribute, in order to decide the boundaries of k-bins 
	min_and_max = []
	np_full_dataset = np.array(full_dataset)
	attr_length = len(full_dataset.schema) - 2
	min_and_max = np.zeros((attr_length, 2))
	for i in range(1, attr_length + 1):
		if(full_dataset.schema[i].type == "CONTINUOUS"):
			row = np_full_dataset[:,i].astype(float)
			max = np.amax(row)
			min = np.amin(row)
			min_and_max[i-1][0]=min
			min_and_max[i-1][1]=max
	min_and_max = np.transpose(min_and_max)
	
	#Build models
	if(ENABLE_VAL == 1):
		label_ratio, save_all_prob, save_all_threshold = Naive_Bayes.showme_dataset(full_dataset, NUM_BINS, M, min_and_max)
		accuracy, precision, recall = compute_test_results(label_ratio, save_all_prob, full_dataset)
		ROC_area = compute_ROC_area()
		print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
	elif(ENABLE_VAL == 0):
		datasets = fold_5_cv(full_dataset)
		accuracies, precisions, recalls = naive_bayes_cv(datasets, min_and_max)
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
		
		
def naive_bayes_cv(datasets, min_and_max):
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
		label_ratio, save_all_prob, save_all_threshold = Naive_Bayes.showme_dataset(train_data, NUM_BINS, M, min_and_max)
		accuracy, precision, recall = compute_test_results(label_ratio, save_all_prob, val_data)
		accuracies.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
		print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
		
	return accuracies, precisions, recalls
		
	
def fold_5_cv(full_dataset):		#Divide full_dataset into stratified 5-folds
	#Separate the full_dataset into two sets in terms of the label
	true_set = mldata.ExampleSet(ex for ex in full_dataset if ex[-1] == True)
	false_set = mldata.ExampleSet(ex for ex in full_dataset if ex[-1] == False)
	shuffle(true_set)
	shuffle(false_set)
	
	#Calculate the length of each set
	true_len = len(true_set)
	true_len_part = true_len / 5
	false_len = len(false_set)
	false_len_part = false_len / 5
	
	datasets = []
	
	for i in range(5):
		dataset = mldata.ExampleSet()
		for j in range(int(i * true_len_part), int((i + 1) * true_len_part)):
			dataset.append(true_set[j])
		for j in range(int(i * false_len_part), int((i + 1) * false_len_part)):
			dataset.append(false_set[j])
		datasets.append(dataset)

	return datasets
	

def compute_test_results(label_ratio, save_all_prob, test_dataset):
	global ROC_matrix
	global err_rates
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	for data in test_dataset:
		pred = Naive_Bayes.showme_example(data, save_all_prob, label_ratio)
		label = data[-1]
		ROC_matrix.append([pred, label])
		if(label == 1):
			#if(label == 1):
			#	true_positive += 1
			#else:
			#	false_negative += 1
			if(pred > 0.5):
				true_positive += 1
			elif(pred < 0.5):
				false_negative += 1
			else:
				random.seed(12345)
				r = random.randint(0, 1)
				if(r == 1):
					true_positive += 1
				else:
					false_negative += 1
		else:
			#if(label == 1):
			#	false_positive += 1
			#else:
			#	true_negative += 1
			if(pred > 0.5):
				false_positive += 1
			elif(pred < 0.5):
				true_negative += 1
			else:
				random.seed(12345)
				r = random.randint(0, 1)
				if(r == 1):
					false_positive += 1
				else:
					true_negative += 1
	accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)
	err_rate = 1 - accuracy
	err_rates.append(err_rate)
	
	return accuracy, precision, recall
	
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
		if(true_negative != 0):
			FP_rate = true_negative / (true_negative + false_positive)
		ROC_value.append([TP_rate, FP_rate])
	ROC_value = sorted(ROC_value, key = itemgetter(1))
	length = len(ROC_value)
	ROC_area = 0
	for i in range(length - 1):
		ROC_area += (ROC_value[i][0] + ROC_value[i + 1][0]) * (ROC_value[i + 1][1] - ROC_value[i][1]) / 2
	return ROC_area
	

def shuffle(datalist):			#shuffle datalist with random seed 12345
	random.seed(12345)
	random.shuffle(datalist)
	
#For classifier_comparison.py
def compute_err_rates():
	if(ENABLE_VAL == 1):
		raise ValueError("ENABLE_VAL should be 0")
	print("Naive Bayes:")
	main()
	return err_rates
	
	
if __name__ == '__main__':
	main()