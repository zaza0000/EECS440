from constants import *
import mldata
import random
import build_tree

def main():
	#Error value processing
	if(ENABLE_VAL != 0 and ENABLE_VAL != 1):
		raise ValueError("ENABLE_VAL should be 0 or 1")
	if(ENABLE_GAIN != 0 and ENABLE_GAIN != 1):
		raise ValueError("ENABLE_GAIN should be 0 or 1")
	if(MAX_DEPTH < 0):
		raise ValueError("MAX_DEPTH should be nonnegative")
	elif(type(MAX_DEPTH) != int):
		raise TypeError("MAX_DEPTH should be an integer")

	#Read data
	path_name = DATA_PATH.rpartition('/')
	path = path_name[0]
	name = path_name[2]
	full_dataset = mldata.parse_c45(name, path)
	
	#Build tree and output all results
	if(ENABLE_VAL == 1):
		tree = build_tree.build_DecisionTree(MAX_DEPTH, EPS, full_dataset, ENABLE_GAIN)
		size = tree.get_tree_size()
		max_depth = tree.get_tree_depth()
		first_feature_index = tree.get_root().get_attriIndex()
		first_feature = full_dataset.schema.features[first_feature_index].name
		acc = tree.classify_dataset(full_dataset)
		print('Accuracy: %.4f\n\nSize: %d\n\nMaximum Depth: %d\n\nFirst Feature: %s' % (acc, size, max_depth, first_feature))
	elif(ENABLE_VAL == 0):
		datasets = fold_5_cv(full_dataset)
		trees, sizes, first_features, accs, max_depths = build_trees(datasets)
		acc_sum = 0
		for i in range(5):
			acc_sum += accs[i]
		acc = acc_sum / 5
		print('\nAverage Accuracy: %.4f' % acc)
		
		
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
		print('Tree %d:\n\nAccuracy: %.4f\n\nSize: %d\n\nMaximum Depth: %d\n\nFirst Feature: %s' % (i + 1, acc, size, max_depth, first_feature))
		
	return trees, sizes, first_features, accs, max_depths
		
	
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
	
	
if __name__ == '__main__':
	main()