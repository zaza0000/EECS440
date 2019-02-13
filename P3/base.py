from constants import *
import mldata
import random
import build_tree
import Naive_Bayes
import logreg
import numpy as np

ROC_matrix = []
err_rates = []

def main():
	global err_rates
	#Error value processing
	if(ENABLE_VAL != 0 and ENABLE_VAL != 1):
		raise ValueError("ENABLE_VAL should be 0 or 1")
	if(ENABLE_GAIN != 0 and ENABLE_GAIN != 1):
		raise ValueError("ENABLE_GAIN should be 0 or 1")
	if(MAX_DEPTH < 0):
		raise ValueError("MAX_DEPTH should be nonnegative")
	elif(type(MAX_DEPTH) != int):
		raise TypeError("MAX_DEPTH should be an integer")
	if(NUM_BINS < 2):
		raise ValueError("NUM_BINS should be greater that 2")
	elif(type(NUM_BINS) != int):
		raise TypeError("NUM_BINS should be an integer")
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
	
	#Build tree and output all results
	if(ENABLE_VAL == 0):
		datasets = fold_5_cv(full_dataset)
		if(ALGORITHM == 1):
			trees, sizes, first_features, accs, max_depths = build_trees(datasets)
			acc_sum = 0
			for i in range(5):
				acc_sum += accs[i]
			acc = acc_sum / 5
			print('Average Accuracy: %.4f\n' % acc)
		elif(ALGORITHM == 2):
			min_and_max = caculate_min_and_max(full_dataset)
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
			print("Accuracy: %.3f %.3f\nPrecision: %.3f %.3f\nRecall: %.3f %.3f\n" % (avg_accuracy, std_accuracy, avg_precision, std_precision, avg_recall, std_recall))
		elif(ALGORITHM == 3):
			err_rates = logreg.compute_err_rates()

			
def caculate_min_and_max(full_dataset):
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
	return min_and_max
		
		
def build_trees(datasets):			#Build a tree using datasets as training data
	#Initialize lists to save outputs
	trees = []
	sizes = []
	max_depths = []
	first_features = []
	accs = []
	
	#Build each tree and output results
	for i in range(5):
		train_data = mldata.ExampleSet()
		for j in range(1, 5):
			for index in range(len(datasets[(i + j) % 5])):
				train_data.append(datasets[(i + j) % 5][index])
		val_data = datasets[i]
		shuffle(train_data)
		shuffle(val_data)
		tree = build_tree.build_DecisionTree(MAX_DEPTH, EPS, train_data, ENABLE_GAIN)
		size = tree.get_tree_size()
		max_depth = tree.get_tree_depth()
		trees.append(tree)
		sizes.append(size)
		max_depths.append(max_depth)
		first_feature_index = tree.get_root().get_attriIndex()
		first_feature = train_data.schema.features[first_feature_index].name
		first_features.append(first_feature)
		acc = tree.classify_dataset(val_data)
		accs.append(acc)
		err_rates.append(1 - acc)
		print('Tree %d:\nAccuracy: %.4f\nSize: %d\nMaximum Depth: %d\nFirst Feature: %s\n' % (i + 1, acc, size, max_depth, first_feature))
		
	return trees, sizes, first_features, accs, max_depths
	
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
	

def shuffle(datalist):			#shuffle datalist with random seed 12345
	random.seed(12345)
	random.shuffle(datalist)
	
def compute_err_rates():
	if(ENABLE_VAL == 1):
		raise ValueError("ENABLE_VAL should be 0")
	print("Base:")
	main()
	return err_rates
	
	
if __name__ == '__main__':
	main()