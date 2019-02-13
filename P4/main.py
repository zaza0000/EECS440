from constants import *
import mldata
import random
import build_tree_boosting
import naive_gayes
import IG_v5_boosting
import logreg
import numpy as np
from operator import itemgetter

ROC_matrix = []
err_rates = []

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
	path_name = DATA_PATH.rpartition('/')
	path = path_name[0]
	name = path_name[2]
	full_dataset = mldata.parse_c45(name, path)
	
	#Build models
	if(ENABLE_VAL == 1):
		if(ALGORITHM == 1):
			weight = 1 / len(full_dataset) * np.ones(len(full_dataset))
			weight = weight.reshape(-1,1)
			alpha_list, label_list = build_tree_boosting.boosting(MAX_DEPTH, EPS, full_dataset, full_dataset, ENABLE_GAIN, ITER, weight)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(full_dataset, f_list)
			ROC_area = compute_ROC_area()
			print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
		elif(ALGORITHM == 2):
			alpha_list, label_list = naive_gayes.naive_bayes(full_dataset, full_dataset, ITER, NUM_BINS, M)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(full_dataset, f_list)
			ROC_area = compute_ROC_area()
			print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
		elif(ALGORITHM == 3):
			lg = logreg.Logistic_Regression(lambdaa = LAMBDA, training_data = full_dataset, iteration = 1, learning_rate = LR, boosting = True)
			lg, alpha_list, label_list = update_lg(lg, full_dataset)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(full_dataset, f_list)
			ROC_area = compute_ROC_area()
			print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
	elif(ENABLE_VAL == 0):
		datasets = fold_cv(full_dataset, 5)
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
		
		
def update_lg(lg, test_dataset):
	alpha_list = []
	label_list = []
	for i in range(ITER2):
		prediction, true_label = lg.classify_data(test_dataset)
		label_list.append(prediction)
		flag, alpha = lg.updata_boosting_w()
		alpha_list.append(alpha)
		if(flag == False or i == ITER - 1):
			break
		lg.train_weights_oneMore_iter()
	label_list = np.array(label_list)
	label_list = label_list.T

	return lg, alpha_list, label_list
		
def cv(datasets):
	accuracies = []
	precisions = []
	recalls = []
	length = len(datasets)
	for i in range(length):
		train_data = mldata.ExampleSet()
		for j in range(1, length):
			for index in range(len(datasets[(i + j) % length])):
				train_data.append(datasets[(i + j) % length][index])
		val_data = datasets[i]
		shuffle(train_data)
		shuffle(val_data)
		train_data_with_val = fold_cv(train_data, 3)
		if(P != 0):
			for data in train_data:
				if(random.random() <= P):
					if(data[-1] == True):
						data[-1] = False
					elif(data[-1] == False):
						data[-1] = True
		if(ALGORITHM == 1):
			weight = 1 / len(train_data) * np.ones(len(train_data))
			weight = weight.reshape(-1,1)
			alpha_list, label_list = build_tree_boosting.boosting(MAX_DEPTH, EPS, train_data, val_data, ENABLE_GAIN, ITER, weight)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(val_data, f_list)
			accuracies.append(accuracy)
			precisions.append(precision)
			recalls.append(recall)
			ROC_area = compute_ROC_area()
			print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nArea under ROC: %.3f\n" % (accuracy, precision, recall, ROC_area))
		elif(ALGORITHM == 2):
			alpha_list, label_list = naive_gayes.naive_bayes(train_data, val_data, ITER, NUM_BINS, M)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(val_data, f_list)
			accuracies.append(accuracy)
			precisions.append(precision)
			recalls.append(recall)
			print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
		elif(ALGORITHM == 3):
			lg = logreg.Logistic_Regression(lambdaa = LAMBDA, training_data = train_data, iteration = 100, learning_rate = LR, boosting = True)
			lg, alpha_list, label_list = update_lg(lg, val_data)
			f_list = compute_f_list(alpha_list, label_list)
			accuracy, precision, recall = compute_test_results(val_data, f_list)
			accuracies.append(accuracy)
			precisions.append(precision)
			recalls.append(recall)
			print("Classifier %d:\nAccuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\n" % (i + 1, accuracy, precision, recall))
		
	return accuracies, precisions, recalls
		
	
def fold_cv(full_dataset, num_folds):		#Divide full_dataset into stratified 5-folds
	#Separate the full_dataset into two sets in terms of the label
	true_set = mldata.ExampleSet(ex for ex in full_dataset if ex[-1] == True)
	false_set = mldata.ExampleSet(ex for ex in full_dataset if ex[-1] == False)
	shuffle(true_set)
	shuffle(false_set)
	
	#Calculate the length of each set
	true_len = len(true_set)
	true_len_part = true_len / num_folds
	false_len = len(false_set)
	false_len_part = false_len / num_folds
	
	datasets = []
	
	for i in range(num_folds):
		dataset = mldata.ExampleSet()
		for j in range(int(i * true_len_part), int((i + 1) * true_len_part)):
			dataset.append(true_set[j])
		for j in range(int(i * false_len_part), int((i + 1) * false_len_part)):
			dataset.append(false_set[j])
		datasets.append(dataset)

	return datasets
	
def compute_f_list(alpha_list, label_list):
	sum_alpha = 0
	f_list = []
	for alpha in alpha_list:
		sum_alpha += alpha
	for i in range(len(label_list)):
		f = 0
		for t in range(len(alpha_list)):
			f += alpha_list[t] / sum_alpha * label_list[i][t]
		f_list.append(f)
	return f_list
	

def compute_test_results(test_dataset, pred_list):
	global ROC_matrix
	global err_rates
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	for i in range(len(test_dataset)):
		label = test_dataset[i][-1]
		pred = pred_list[i]
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
	if(true_positive + false_positive == 0):
		precision = 0
	else:
		precision = true_positive / (true_positive + false_positive)
	if(true_positive + false_negative == 0):
		recall = 0
	else:
		recall = true_positive / (true_positive + false_negative)
	err_rates.append(1 - accuracy)
	
	return accuracy, precision, recall
	

#compute ROC w.r.t every instance
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
	

def shuffle(datalist):			#shuffle datalist with random seed 12345
	random.seed(12345)
	random.shuffle(datalist)
	
#For classifier_comparision.py
def compute_err_rates():
	if(ENABLE_VAL == 1):
		raise ValueError("ENABLE_VAL should be 0")
	print("Boosting")
	main()
	return err_rates
	
	
if __name__ == '__main__':
	main()