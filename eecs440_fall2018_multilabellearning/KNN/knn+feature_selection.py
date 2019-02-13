import random
import numpy as np
from data_preprocess import Exampleset
from constants import *

############################################
# class KNN:
#	The main class of KNN algorithm
############################################
class KNN():
	def __init__(self, dataset_name = None, k = 5, s = 1):
		print("Load data...", DATASET_NAME)
		self.full_dataset = Exampleset(dataset_name)
		if(self.full_dataset == None):
			return None
		self.k = k
		self.s = s
		self.k_nearest_neighbor = None
		self.testing_set = None
		self.training_set = None
		self.dis = None
		self.dis2 = None
		self.Nx = None
		self.real_test = None
		self.label_num = self.full_dataset.num_of_labels
		self.PH_1 = np.zeros(self.full_dataset.num_of_labels)
		self.PH_0 = np.zeros(self.full_dataset.num_of_labels)
		self.PEH_1 = np.zeros([self.full_dataset.num_of_labels, self.k + 1])
		self.PEH_0 = np.zeros([self.full_dataset.num_of_labels, self.k + 1])
		print("Cross validation:")
		self.Cross_validation()

	############################################
	# Cross_validation():
	#	Use CV to evaluate the performance of KNN
	############################################
	def Cross_validation(self, n = 5):
		full_dataset = self.full_dataset.dataset
		shuffle(full_dataset)
		new_data_set = np.array_split(full_dataset, n)		# after shuffle and split
		loss_list = list()
		err_list = list()
		acc_list = list()
		pre_list = list()
		recall_list = list()
		f1_list = list()
		avg_loss = 0
		avg_err = 0
		avg_acc = 0
		avg_pre = 0
		avg_recall = 0
		avg_f1 = 0
		for i in range(n):
			print("  Classifier ", i+1,":")
			self.testing_set = new_data_set[i]
			self.training_set = new_data_set[(i+1)%n]
			for j in range(n-2):
				self.training_set = np.r_[self.training_set, new_data_set[(i+2+j)%n]]
			self.Train()
			prediction, index_array = self.Test()
			loss = self.Hamming_loss(prediction)
			Oerror = self.One_error(index_array)
			acc = self.Accuracy_score(prediction)
			preci = self.Precision(prediction)
			recall = self.Recall(prediction)
			F1 = self.F1_score(recall, preci)
			avg_loss = avg_loss + loss
			avg_err = avg_err + Oerror
			avg_acc = avg_acc + acc
			avg_pre = avg_pre + preci
			avg_recall = avg_recall + recall
			avg_f1 = avg_f1 + F1
			loss_list.append(loss)
			err_list.append(Oerror)
			acc_list.append(acc)
			pre_list.append(preci)
			recall_list.append(recall)
			f1_list.append(F1)
			print("	   Hamming acc: ",1 - loss)
			print("	   One error: ",Oerror)
			print("	   Accuracy Score: ",acc)
			print("	   Precison: ",preci)
			print("	   Recall: ",recall)
			print("	   F1 Score: ", F1)
		avg_loss = avg_loss/n
		avg_err = avg_err/n
		avg_acc = avg_acc/n
		avg_pre = avg_pre/n
		avg_recall = avg_recall/n
		avg_f1 = avg_f1/n
		std_loss = 0
		std_err = 0
		std_acc = 0
		std_pre = 0
		std_recall = 0
		std_f1 = 0
		for i in range(n):
			std_loss += (loss_list[i] - avg_loss) ** 2
			std_err += (loss_list[i] - avg_loss) ** 2
			std_acc += (acc_list[i] - avg_acc) ** 2
			std_pre += (pre_list[i] - avg_pre) ** 2
			std_recall += (recall_list[i] - avg_recall) ** 2
			std_f1 += (f1_list[i] - avg_f1) ** 2
		std_loss = (std_loss / n) ** 0.5
		std_err = (std_err / n) ** 0.5
		std_acc = (std_acc / n) ** 0.5
		std_pre = (std_pre / n) ** 0.5	
		std_recall = (std_recall / n) ** 0.5
		std_f1 = (std_f1 / n) ** 0.5
		print()
		print("avg_hacc std: %.4f %.4f\n" % (1-avg_loss, 1-std_loss))
		print("avg_err std: %.4f %.4f\n" % (avg_err, std_err))
		print("avg_acc std: %.4f %.4f\n" % (avg_acc, std_acc))
		print("avg_pre std: %.4f %.4f\n" % (avg_pre, std_pre))
		print("avg_recall std: %.4f %.4f\n" % (avg_recall, std_recall))
		print("avg_f1 std: %.4f %.4f\n" % (avg_f1, std_f1))


	############################################
	# Find_knn():
	#	Find k nearnest neighbor for all the data
	############################################
	def Find_knn(self):
		num_of_data = len(self.training_set)
		Nx = np.zeros([num_of_data, self.k])
		for i in range(num_of_data):
			Nx[i] = self.Find_knn_sub(i)

		return Nx

	############################################
	# Find_knn_sub():
	#	Find k nearnest neighbor for data[index]
	############################################
	def Find_knn_sub(self, index):
		num_of_data= len(self.training_set)
		neighbors = np.zeros(self.k)
		train_x = self.training_set[: ,-(self.full_dataset.num_of_labels):]
		for i in range(num_of_data - index):
			self.dis[index][i+index] = ((train_x[i] - train_x[index]) ** 2).sum()
			self.dis[i+index][index] = self.dis[index][i+index]

		for i in range(self.k):
			temp_dis = float('inf')
			temp_index = 0
			for j in range(num_of_data):
				if(j != index and self.dis[index][j] < temp_dis):
					temp_dis = self.dis[index][j]
					temp_index = j
			self.dis[index][temp_index] = float('inf')
			neighbors[i] = int(temp_index)

		return neighbors

	############################################
	# Train():
	#	Gain all the probs from training data
	############################################
	def Train(self):
		train_data_num = len(self.training_set)
		self.dis = np.zeros([train_data_num, train_data_num])
		train_x = self.training_set[:, : -(self.full_dataset.num_of_labels)]
		train_y = self.training_set[:, -(self.full_dataset.num_of_labels):]
		test_x = self.testing_set[: ,-(self.full_dataset.num_of_labels):]
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		# prior probs
		for i in range(self.full_dataset.num_of_labels):
			num_of_label_i = 0
			for j in range(train_data_num):
				if train_y[j][i] == 1:
					num_of_label_i = num_of_label_i + 1
			self.PH_1[i] = (self.s + num_of_label_i) / (self.s * 2 + train_data_num)
			self.PH_0[i] = 1 - self.PH_1[i]

		selected_features = list()
		feature_set = dict()
		is_improved = 1
		max_feature_num = len(train_x)-2
		for i in range(max_feature_num):
			feature_set[i+1] = 0

		real_train = np.array(training_set[:,0])
		self.real_test = self.testing_set[:, -(self.full_dataset.num_of_labels):]

		pre_error_rate = -1
		while(len(selected_features) != min(NUM_OF_FEATURES, max_feature_num) and is_improved == 1):
			feature_set = three_fold_knn(feature_set, train_x, selected_features)
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


		# find knn
		self.Nx = self.Find_knn()
		# posterior probs
		for i in range(self.full_dataset.num_of_labels):
			C_1 = np.zeros(self.k + 1)
			C_0 = np.zeros(self.k + 1)
			for j in range(train_data_num):
				sigma = 0
				N = self.Nx[j]
				for k in range(self.k):
					l = int(N[k])
					sigma = sigma + int(train_y[l][i])
					if(train_y[j][i] == 1):
						C_1[sigma] = C_1[sigma] + 1
					else:
						C_0[sigma] = C_0[sigma] + 1
			for j in range(self.k + 1):
				self.PEH_1[i][j] = (self.s + C_1[j]) / (self.s * (self.k + 1) + np.sum(C_1))
				self.PEH_0[i][j] = (self.s + C_0[j]) / (self.s * (self.k + 1) + np.sum(C_0))

	############################################
	# Find_knn_test():
	#	Find k nearnest neighbor for all testing data
	############################################
	def Find_knn_test(self):
		num_of_data = len(self.testing_set)
		Nx = np.zeros([num_of_data, self.k])
		for i in range(num_of_data):
			Nx[i] = self.Find_knn_sub_test(i)

		return Nx

	############################################
	# Find_knn_sub_test():
	#	Find k nearnest neighbor for test_data[index]
	############################################
	def Find_knn_sub_test(self, index):
		num_of_data= len(self.training_set)
		neighbors = np.zeros(self.k)
		train_x = self.training_set[: ,-(self.full_dataset.num_of_labels):]
		test_x = self.testing_set[: ,-(self.full_dataset.num_of_labels):]
		for i in range(num_of_data):
			self.dis2[index][i] = ((train_x[i] - test_x[index]) ** 2).sum()
		for i in range(self.k):
			temp_dis = float('inf')
			temp_index = 0
			for j in range(num_of_data):
				if(self.dis2[index][j] < temp_dis):
					temp_dis = self.dis2[index][j]
					temp_index = j
			self.dis2[index][temp_index] = float('inf')
			neighbors[i] = int(temp_index)

		return neighbors


	def Test(self):
		# Computing ⃗y_t
		train_x = self.training_set[:, : -(self.full_dataset.num_of_labels)]
		train_y = self.training_set[:, -(self.full_dataset.num_of_labels):]
		test_x = self.testing_set[:, : -(self.full_dataset.num_of_labels)]
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		predict = np.zeros(test_y.shape)
		num_of_data = len(test_x)
		self.dis2 = np.zeros([len(test_x), len(train_x)])
		# Identify N(t)
		test_neighbors = self.Find_knn_test()
		max_p_array = list()

		for i in range(num_of_data):
			max_p_index = -1
			max_p_value = 0
			for j in range(self.full_dataset.num_of_labels):
				index_num = j
				num_of_label_i = 0
				for test_nei in test_neighbors[i]:
					index = int(test_nei)
					num_of_label_i = num_of_label_i + int(train_y[index][j])
				# argmax
				label_1 = self.PH_1[j] * self.PEH_1[j][num_of_label_i]
				label_0 = self.PH_0[j] * self.PEH_0[j][num_of_label_i]
				if(label_1 > label_0):
					predict[i][j] = 1
					if(label_1 - label_0 > max_p_value):
						max_p_value = label_1 - label_0
						max_p_index = index_num
			max_p_array.append(max_p_index)
		return predict, max_p_array

	############################################
	# Hamming_loss():
	#	Using Hamming_loss to do the evaluation
	############################################
	def Hamming_loss(self, predict):
		testing_set_num = len(self.testing_set)
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		temp = 0
		for i in range(testing_set_num):
			for j in range(len(test_y[0])):
				if(test_y[i][j] != predict[i][j]):
					temp = temp + 1
		loss = temp / self.full_dataset.num_of_labels / testing_set_num

		return loss

	############################################
	# One_error():
	#	Using One_error to do the evaluation
	############################################
	def One_error(self, index_array):
		testing_set_num = len(self.testing_set)
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		temp = 0
		for i in range(testing_set_num):
			index = index_array[i]
			if(index != -1):
				if(test_y[i][index] == 0):
					temp = temp + 1
		error = temp / testing_set_num
		return error

	############################################
	# Accuracy():
	#	Using Accuracy to do the evaluation
	############################################
	def Accuracy_score(self, predict):
		testing_set_num = len(self.testing_set)
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		temp = 0
		for i in range(testing_set_num):
			numerator = 1
			denominator = 1
			for j in range(len(test_y[0])):
				if(test_y[i][j] != predict[i][j]):
					numerator = 0
					break
			temp = temp + numerator / denominator
		acc = temp / testing_set_num

		return acc

	############################################
	# Precision():
	#	Using Precision to do the evaluation
	############################################
	def Precision(self, predict):
		testing_set_num = len(self.testing_set)
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		temp = 0
		for i in range(testing_set_num):
			numerator = 0
			denominator = 0
			for j in range(len(test_y[0])):
				if(predict[i][j] == 1):
					if(test_y[i][j] == predict[i][j]):
						numerator = numerator + 1
					denominator = denominator + 1
			if(denominator != 0):
				temp = temp + numerator / denominator
		precision = temp / testing_set_num

		return precision

	############################################
	# Recall():
	#	Using Recall to do the evaluation
	############################################
	def Recall(self, predict):
		testing_set_num = len(self.testing_set)
		test_y = self.testing_set[:, -(self.full_dataset.num_of_labels):]
		temp = 0
		for i in range(testing_set_num):
			numerator = 0
			denominator = 0
			for j in range(len(test_y[0])):
				if(test_y[i][j] == 1):
					if(test_y[i][j] == predict[i][j]):
						numerator = numerator + 1
					denominator = denominator + 1
			if(denominator != 0):
				temp = temp + numerator / denominator
		recall = temp / testing_set_num

		return recall

	############################################
	# F1_score():
	#	Using F1_score to do the evaluation
	############################################
	def F1_score(self, recall, precision):
		F1 = 0
		if(recall != 0 and precision != 0):
			F1 = 2*recall*precision/(recall + precision)

		return F1

def three_fold_knn(feature_set, training_data, real_train_set, schema):
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

def shuffle(datalist):			#shuffle datalist with random seed 12345
	random.seed(12345)
	random.shuffle(datalist)



def main():
	model = KNN(DATASET_NAME, K, S)


if __name__ == "__main__":
	main()