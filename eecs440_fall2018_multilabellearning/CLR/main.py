import data_processing
from constants import *
import D_matrix_generate
import Bayes
import random
import cross_validation
import math

def main():
	#attr_type_flag, full_labels, data_value_list, data_label_list = data_processing.data_processing(DATASET_NAME, 0)
	#Djk_label_list, Djv_label_list = D_matrix_generate.D_matrix_generate(full_labels, data_label_list)
	#model, max_and_min = Bayes.Naive_Bayes(attr_type_flag, data_value_list, full_labels, Djk_label_list, Djv_label_list)
	#attr_type_flag, full_labels, data_value_list, data_label_list = data_processing.data_processing(DATASET_NAME, 0)
	#accuracy = Bayes.classify(model, max_and_min, attr_type_flag, data_value_list, full_labels, data_label_list)
	#return
	attr_type_flag, full_labels, data_value_list, data_label_list = data_processing.data_processing(DATASET_NAME, 0)
	max_and_min = {}
	for i in range(len(attr_type_flag)):
		if(attr_type_flag[str(i)] == "1"):
			for data in data_value_list:
				if data.__contains__(str(i)):
					value = float(data[str(i)])
					max = value
					min = value
					break
			for data in data_value_list:
				if data.__contains__(str(i)):
					value = float(data[str(i)])
					if value > max:
						max = value
					if value < min:
						min = value
					max_and_min["%s_max" % str(i)] = str(max)
					max_and_min["%s_min" % str(i)] = str(min)
	print(max_and_min)
	minus = float(max_and_min["102_max"]) - float(max_and_min["102_min"])
	print(minus)
	datasets_value, datasets_label = cross_validation.construct_cv_folds(N_FOLDS, data_value_list, data_label_list)
	#print(datasets_label)
	Accuracy_scores = []
	HamLosses = []
	Precisions = []
	Recalls = []
	F1s = []
	AVGPRECs = []
	RANKLOSSes = []
	avg_Accuracy_score = 0
	avg_HamLoss = 0
	avg_Precision = 0
	avg_Recall = 0
	avg_F1 = 0
	avg_AVGPREC = 0
	avg_RANKLOSS = 0
	
	std_Accuracy_score = 0
	std_HamLoss = 0
	std_Precision = 0
	std_Recall = 0
	std_F1 = 0
	std_AVGPREC = 0
	std_RANKLOSS = 0
	for i in range(N_FOLDS):
		training_set_value = []
		training_set_label = []
		for j in range(N_FOLDS):
			if i != j:
				training_set_value = training_set_value + datasets_value[j]
				training_set_label = training_set_label + datasets_label[j]
		testing_set_value = datasets_value[i]
		testing_set_label = datasets_label[i]
		Djk_label_list, Djv_label_list = D_matrix_generate.D_matrix_generate(full_labels, training_set_label)
		model, max_and_min = Bayes.Naive_Bayes(attr_type_flag, training_set_value, full_labels, Djk_label_list, Djv_label_list, max_and_min)
		Accuracy_score, HamLoss, Precision, Recall, F1, AVGPREC, RANKLOSS = Bayes.classify(model, max_and_min, attr_type_flag, testing_set_value, full_labels, testing_set_label)
		Accuracy_scores.append(Accuracy_score)
		HamLosses.append(HamLoss)
		Precisions.append(Precision)
		Recalls.append(Recall)
		F1s.append(F1)
		AVGPRECs.append(AVGPREC)
		RANKLOSSes.append(RANKLOSS)
		avg_Accuracy_score += Accuracy_score
		avg_HamLoss += HamLoss
		avg_Precision += Precision
		avg_Recall += Recall
		avg_F1 += F1
		avg_AVGPREC += AVGPREC
		avg_RANKLOSS += RANKLOSS
	avg_Accuracy_score /= N_FOLDS
	avg_HamLoss /= N_FOLDS
	avg_Precision /= N_FOLDS
	avg_Recall /= N_FOLDS
	avg_F1 /= N_FOLDS
	avg_AVGPREC /= N_FOLDS
	avg_RANKLOSS /= N_FOLDS
	for i in range(N_FOLDS):
		std_Accuracy_score += (Accuracy_scores[i] - avg_Accuracy_score) ** 2
		std_HamLoss += (HamLosses[i] - avg_HamLoss) ** 2
		std_Precision += (Precisions[i] - avg_Precision) ** 2
		std_Recall += (Recalls[i] - avg_Recall) ** 2
		std_F1 += (F1s[i] - avg_F1) ** 2
		std_AVGPREC += (AVGPRECs[i] - avg_AVGPREC) ** 2
		std_RANKLOSS += (RANKLOSSes[i] - avg_RANKLOSS) ** 2
	std_Accuracy_score = math.sqrt(std_Accuracy_score / N_FOLDS)
	std_HamLoss = math.sqrt(std_HamLoss / N_FOLDS)
	std_Precision = math.sqrt(std_Precision / N_FOLDS)
	std_Recall = math.sqrt(std_Recall / N_FOLDS)
	std_F1 = math.sqrt(std_F1 / N_FOLDS)
	std_AVGPREC = math.sqrt(std_AVGPREC / N_FOLDS)
	std_RANKLOSS = math.sqrt(std_RANKLOSS / N_FOLDS)
	print("AVG:\nAccuracy_score=%.4f %.4f\nHamLoss=%.4f %.4f\nPrecision=%.4f %.4f\nRecall=%.4f %.4f\nF1=%.4f %.4f\nAVGPREC=%.4f %.4f\nRANKLOSS=%.4f %.4f" % (avg_Accuracy_score, std_Accuracy_score, avg_HamLoss, std_HamLoss, avg_Precision, std_Precision, avg_Recall, std_Recall, avg_F1, std_F1, avg_AVGPREC, std_AVGPREC, avg_RANKLOSS, std_RANKLOSS))

	
if __name__ == "__main__":
	main()