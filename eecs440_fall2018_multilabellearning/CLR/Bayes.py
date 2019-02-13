from constants import *
import random
import math


def Naive_Bayes(attr_type_flag, data_value_list, full_labels, Djk_label_list, Djv_label_list, max_and_min):
	posibilities, Y_name_list = init_posibilities(attr_type_flag, data_value_list, full_labels)
	num_datas = len(data_value_list)
	num_attributes = len(attr_type_flag)
	for i in range(num_datas):
		data = data_value_list[i]
		Djk_label = Djk_label_list[i]
		Djv_label = Djv_label_list[i]
		for Djk in Djk_label:
			if(posibilities.__contains__("%s=%s" % (Djk, Djk_label[Djk])) == False):
				print("Error1")
				return
			posibilities["%s=%s" % (Djk, Djk_label[Djk])] = str(int(posibilities["%s=%s" % (Djk, Djk_label[Djk])]) + 1)
		for Djv in Djv_label:
			if(posibilities.__contains__("%s=%s" % (Djv, Djv_label[Djv])) == False):
				print("Error2")
				return
			posibilities["%s=%s" % (Djv, Djv_label[Djv])] = str(int(posibilities["%s=%s" % (Djv, Djv_label[Djv])]) + 1)
		
		for attr_num in range(num_attributes):#data
			attr_num = str(attr_num)
			if data.__contains__(attr_num) == False:
				value = "0"
			else:
				value = data[attr_num]
			
			if attr_type_flag[attr_num] == "0":
				for Djk in Djk_label:
					if(posibilities.__contains__("%s=%s|" % (attr_num, value) + Djk + "=" + Djk_label[Djk]) == False):
						print("Error3")
						return
					posibilities["%s=%s|" % (attr_num, value) + Djk + "=" + Djk_label[Djk]] = str(int(posibilities["%s=%s|" % (attr_num, value) + Djk + "=" + Djk_label[Djk]]) + 1)
				
				for Djv in Djv_label:
					if(posibilities.__contains__("%s=%s|" % (attr_num, value) + Djv + "=" + Djv_label[Djv]) == False):
						print("Error3")
						return
					posibilities["%s=%s|" % (attr_num, value) + Djv + "=" + Djv_label[Djv]] = str(int(posibilities["%s=%s|" % (attr_num, value) + Djv + "=" + Djv_label[Djv]]) + 1)

			else:
				min = float(max_and_min["%s_min" % attr_num])
				max = float(max_and_min["%s_max" % attr_num])
				one_piece = (max - min) / NUM_BINS
				part = NUM_BINS - 1
				for num in range(1, NUM_BINS):
					if float(value) < num * one_piece:
						part = num - 1
						break
				for Djk in Djk_label:
					if(posibilities.__contains__("%s=%s|" % (attr_num, str(part)) + Djk + "=" + Djk_label[Djk]) == False):
						print("Error4")
						return
					posibilities["%s=%s|" % (attr_num, str(part)) + Djk + "=" + Djk_label[Djk]] = str(int(posibilities["%s=%s|" % (attr_num, str(part)) + Djk + "=" + Djk_label[Djk]]) + 1)
				for Djv in Djv_label:
					if(posibilities.__contains__("%s=%s|" % (attr_num, str(part)) + Djv + "=" + Djv_label[Djv]) == False):
						print("Error5")
						return
					posibilities["%s=%s|" % (attr_num, str(part)) + Djv + "=" + Djv_label[Djv]] = str(int(posibilities["%s=%s|" % (attr_num, str(part)) + Djv + "=" + Djv_label[Djv]]) + 1)

				
				
	num_labels = len(full_labels)
	for i in range(num_labels - 1):
		pos_posibility = int(posibilities["%s_v=1" % str(i)])
		neg_posibility = int(posibilities["%s_v=0" % str(i)])
		if pos_posibility == 0 and neg_posibility == 0:
			posibilities["%s_v=1" % str(i)] = "0"
			posibilities["%s_v=0" % str(i)] = "0"
		else:
			posibilities["%s_v=1" % str(i)] = str(pos_posibility / (pos_posibility + neg_posibility))
			posibilities["%s_v=0" % str(i)] = str(neg_posibility / (pos_posibility + neg_posibility))
		for j in range(i + 1, num_labels):
			pos_posibility = int(posibilities["%s_%s=1" % (str(i), str(j))])
			neg_posibility = int(posibilities["%s_%s=0" % (str(i), str(j))])
			if pos_posibility == 0 and neg_posibility == 0:
				posibilities["%s_%s=1" % (str(i), str(j))] = "0"
				posibilities["%s_%s=0" % (str(i), str(j))] = "0"
			else:
				posibilities["%s_%s=1" % (str(i), str(j))] = str(pos_posibility / (pos_posibility + neg_posibility))
				posibilities["%s_%s=0" % (str(i), str(j))] = str(neg_posibility / (pos_posibility + neg_posibility))
	pos_posibility = int(posibilities["%s_v=1" % str(num_labels - 1)])
	neg_posibility = int(posibilities["%s_v=0" % str(num_labels - 1)])
	if pos_posibility == 0 and neg_posibility == 0:
		posibilities["%s_v=1" % str(num_labels - 1)] = "0"
		posibilities["%s_v=0" % str(num_labels - 1)] = "0"
	else:
		posibilities["%s_v=1" % str(num_labels - 1)] = str(pos_posibility / (pos_posibility + neg_posibility))
		posibilities["%s_v=0" % str(num_labels - 1)] = str(neg_posibility / (pos_posibility + neg_posibility))
	
	for i in range(num_attributes):
		if(attr_type_flag[str(i)] == "0"):
			p = 0.5
			if M < 0:
				m = 2
			else:
				m = M
			for Y_name in Y_name_list:
				pos_posibility = int(posibilities["%s=1|" % i + Y_name])
				neg_posibility = int(posibilities["%s=0|" % i + Y_name])
				if pos_posibility == 0 and neg_posibility == 0 and m == 0:
					posibilities["%s=1|" % i + Y_name] = "0"
					posibilities["%s=0|" % i + Y_name] = "0"
				else:
					posibilities["%s=1|" % i + Y_name] = str((pos_posibility + m * p) / (pos_posibility + neg_posibility + m))
					posibilities["%s=0|" % i + Y_name] = str((neg_posibility + m * p) / (pos_posibility + neg_posibility + m))
		else:
			if M < 0:
				m = NUM_BINS
			else:
				m = M
			for Y_name in Y_name_list:
				K_bins_posibilities = []
				sum = 0
				for j in range(NUM_BINS):
					value = int(posibilities["%s=%s|" % (str(i), str(j)) +Y_name])
					K_bins_posibilities.append(value)
					sum += value
				for j in range(NUM_BINS):
					if sum == 0 and m == 0:
						posibilities["%s=%s|" % (str(i), str(j)) +Y_name] = "0"
					else:
						posibilities["%s=%s|" % (str(i), str(j)) +Y_name] = str((K_bins_posibilities[j] + m * 1 / NUM_BINS) / (sum + m))
				
	return posibilities, max_and_min
	

def init_posibilities(attr_type_flag, data_value_list, full_labels):
	num_datas = len(data_value_list)
	num_labels = len(full_labels)
	num_attributes = len(attr_type_flag)
	#num_Djk = num_labels * (num_labels - 1) / 2
	#num_Djv = num_labels
	posibilities = {}
	Y_name_list = []
	for i in range(num_labels - 1):
		posibilities["%s_v=1" % str(i)] = "0"
		Y_name_list.append("%s_v=1" % str(i))
		posibilities["%s_v=0" % str(i)] = "0"
		Y_name_list.append("%s_v=0" % str(i))
		for j in range(i + 1, num_labels):
			posibilities["%s_%s=1" % (str(i), str(j))] = "0"	#store the number of instance of Djk=+1
			Y_name_list.append("%s_%s=1" % (str(i), str(j)))
			posibilities["%s_%s=0" % (str(i), str(j))] = "0"
			Y_name_list.append("%s_%s=0" % (str(i), str(j)))
	posibilities["%s_v=1" % str(num_labels - 1)] = "0"
	Y_name_list.append("%s_v=1" % str(num_labels - 1))
	posibilities["%s_v=0" % str(num_labels - 1)] = "0"
	Y_name_list.append("%s_v=0" % str(num_labels - 1))
	
	#max_and_min = {}
	#mean_and_var = {}
	for i in range(num_attributes):
		if(attr_type_flag[str(i)] == "0"):
			for Y_name in Y_name_list:
				posibilities["%s=1|" % i + Y_name] = "0"
				posibilities["%s=0|" % i + Y_name] = "0"
		else:
			#add mean and num
			#for m in range(2):
			#	for n in range(num_datas):
			#		for j in range(num_labels - 1):
			#			if Djv_label_list[n].__contains__("%s_v" % j) and Djv_label_list[n]["%s_v" % j] == str(m):
			#				if mean_and_var.__contains__("mean_%s|%s_v=%s" % (i, j, m)):
			#					mean = float(mean_and_var["mean_%s|%s_v=%s" % (i, j, m)])
			#					mean = mean + float(data_value_list[n][str(i)])
			#					mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = str(mean)
			#					num = int(mean_and_var["num_%s|%s_v=%s" % (i, j, m)])
			#					num += 1
			#					mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = str(num)
			#				else:
			#					mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = data_value_list[n][str(i)]
			#					mean_and_var["var_%s|%s_v=%s" % (i, j, m)] = "0"
			#					mean_and_var["num_%s|%s_v=%s" % (i, j, m)] = "1"
			#			for k in range(j + 1, num_labels):
			#				if Djk_label_list[n].__contains__("%s_%s" % (j, k)) and Djk_label_list[n]["%s_%s" % (j, k)] == str(m):
			#					if mean_and_var.__contains__("mean_%s|%s_%s=%s" % (i, j, k, m)):
			#						mean = float(mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)])
			#						mean = mean + float(data_value_list[n][str(i)])
			#						mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)] = str(mean)
			#						num = int(mean_and_var["num_%s|%s_%s=%s" % (i, j, k, m)])
			#						num += 1
			#						mean_and_var["num_%s|%s_%s=%s" % (i, j, k, m)] = str(num)
			#					else:
			#						mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)] = data_value_list[n][str(i)]
			#						mean_and_var["var_%s|%s_%s=%s" % (i, j, k, m)] = "0"
			#						mean_and_var["num_%s|%s_%s=%s" % (i, j, k, m)] = "1"
			#		j = num_labels - 1
			#		if Djv_label_list[n].__contains__("%s_v" % j) and Djv_label_list[n]["%s_v" % j] == str(m):
			#			if mean_and_var.__contains__("mean_%s|%s_v=%s" % (i, j, m)):
			#				mean = float(mean_and_var["mean_%s|%s_v=%s" % (i, j, m)])
			#				mean = mean + float(data_value_list[n][str(i)])
			#				mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = str(mean)
			#				num = int(mean_and_var["num_%s|%s_v=%s" % (i, j, m)])
			#				num += 1
			#				mean_and_var["num_%s|%s_v=%s" % (i, j, m)] = str(num)
			#			else:
			#				mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = data_value_list[n][str(i)]
			#				mean_and_var["var_%s|%s_v=%s" % (i, j, m)] = "0"
			#				mean_and_var["num_%s|%s_v=%s" % (i, j, m)] = "1"
			#compute mean
			#for m in range(2):
			#	for j in range(num_labels - 1):
			#		for k in range(j + 1, num_labels):
			#			mean = float(mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)])
			#			num = int(mean_and_var["num_%s|%s_%s=%s" % (i, j, k, m)])
			#			mean = mean / num
			#			mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)] = str(mean)
			#for m in range(2):
			#	for j in range(num_labels):
			#		mean = float(mean_and_var["mean_%s|%s_v=%s" % (i, j, m)])
			#		num = int(mean_and_var["num_%s|%s_v=%s" % (i, j, m)])
			#		mean = mean / num
			#		mean_and_var["mean_%s|%s_v=%s" % (i, j, m)] = str(mean)
						
			#add var		
			#for m in range(2):
			#	for n in range(num_datas):
			#		for j in range(num_labels - 1):
			#			if Djv_label_list[n].__contains__("%s_v" % j) and Djv_label_list[n]["%s_v" % j] == str(m):
			#				mean = float(mean_and_var["mean_%s|%s_v=%s" % (i, j, m)])
			#				var = float(mean_and_var["var_%s|%s_v=%s" % (i, j, m)])
			#				var += (float(data_value_list[n][str(i)]) - mean) ** 2
			#				mean_and_var["var_%s|%s_v=%s" % (i, j, m)] = str(var)
			#			for k in range(j + 1, num_labels):
			#				if Djk_label_list[n].__contains__("%s_%s" % (j, k)) and Djk_label_list[n]["%s_%s" % (j, k)] == str(m):
			#					mean = float(mean_and_var["mean_%s|%s_%s=%s" % (i, j, k, m)])
			#					var = float(mean_and_var["var_%s|%s_%s=%s" % (i, j, k, m)])
			#					var += (float(data_value_list[n][str(i)]) - mean) ** 2
			#					mean_and_var["var_%s|%s_%s=%s" % (i, j, k, m)] = str(var)
			#		j = num_labels - 1
			#		if Djv_label_list[n].__contains__("%s_v" % j) and Djv_label_list[n]["%s_v" % j] == str(m):
			#				mean = float(mean_and_var["mean_%s|%s_v=%s" % (i, j, m)])
			#				var = float(mean_and_var["var_%s|%s_v=%s" % (i, j, m)])
			#				var += (float(data_value_list[n][str(i)]) - mean) ** 2
			#				mean_and_var["var_%s|%s_v=%s" % (i, j, m)] = str(var)
							
			#compute var
			#for m in range(2):
			#	for j in range(num_labels - 1):
			#		for k in range(j + 1, num_labels):	
			#			var = float(mean_and_var["var_%s|%s_%s=%s" % (i, j, k, m)])
			#			num = int(mean_and_var["num_%s|%s_%s=%s" % (i, j, k, m)])
			#			var = math.sqrt(var / num)
			#			mean_and_var["var_%s|%s_%s=%s" % (i, j, k, m)] = str(var)
			#for m in range(2):
			#	for j in range(num_labels):	
			#		var = float(mean_and_var["var_%s|%s_v=%s" % (i, j, m)])
			#		num = int(mean_and_var["num_%s|%s_v=%s" % (i, j, m)])
			#		var = math.sqrt(var / num)
			#		mean_and_var["var_%s|%s_v=%s" % (i, j, m)] = str(var)

			#for data in data_value_list:
			#	if data.__contains__(str(i)):
			#		value = float(data[str(i)])
			#		max = value
			#		min = value
			#		break
			#for data in data_value_list:
			#	if data.__contains__(str(i)):
			#		value = float(data[str(i)])
			#		if value > max:
			#			max = value
			#		if value < min:
			#			min = value
			#max_and_min["%s_max" % str(i)] = str(max)
			#max_and_min["%s_min" % str(i)] = str(min)
			for Y_name in Y_name_list:
				for j in range(NUM_BINS):
					posibilities["%s=%s|" % (str(i), str(j)) +Y_name] = "0"
	return posibilities, Y_name_list

	
def classify(posibilities, max_and_min, attr_type_flag, data_value_list, full_labels, data_label_list):
	num_labels = len(full_labels)
	num_attributes = len(attr_type_flag)
	num_datas = len(data_value_list)
	RP = 0
	RN = 0
	IP = 0
	IN = 0
	AVGPREC = 0
	RANKLOSS = 0
	mistake = 0
	for num in range(num_datas):
		data = data_value_list[num]
		label = data_label_list[num]
		data_label = {}
		for i in range(num_labels - 1):
			compute_logp(data, i, "v", posibilities, num_attributes, attr_type_flag, max_and_min, data_label)
			for j in range(i + 1, num_labels):
				compute_logp(data, i, j, posibilities, num_attributes, attr_type_flag, max_and_min, data_label)
		compute_logp(data, num_labels - 1, "v", posibilities, num_attributes, attr_type_flag, max_and_min, data_label)
		#print(data)
		#print(label)
		#print(data_label)
		prediction = []
		for i in range(num_labels + 1):
			prediction.append(0)
		for i in range(num_labels - 1):
			if data_label["%s_v" % str(i)] == "1":
				prediction[i] += 1
			else:
				prediction[num_labels] += 1
			for j in range(i + 1, num_labels):
				if data_label["%s_%s" % (str(i), str(j))] == "1":
					prediction[i] += 1
				else:
					prediction[j] += 1
		if data_label["%s_v" % str(num_labels - 1)] == "1":
			prediction[num_labels - 1] += 1
		else:
			prediction[num_labels] += 1
			
	#fully correct
		pred_label = []
		for i in range(len(prediction) - 1):
			if prediction[i] >= prediction[num_labels]:
				pred_label.append(1)
			else:
				pred_label.append(0)
		#print("prediction:",prediction)
		#print("pred_label:",pred_label)
		#print("label:",label)
		for i in range(len(pred_label)):
			if label.__contains__(str(i)):
				if pred_label[i] != int(label[str(i)]):
					mistake += 1
					break
			else:
				if pred_label[i] != 0:
					mistake += 1
					break
		for i in range(len(pred_label)):
			if label.__contains__(str(i)) == False:
				value = 0
			else:
				value = int(label[str(i)])
			if pred_label[i] == 1 and value == 1:
				RP += 1
			elif pred_label[i] == 1 and value == 0:
				IP += 1
			elif pred_label[i] == 0 and value == 1:
				RN += 1
			else:
				IN += 1
		num_pos = 0
		num_neg = 0
		sum_AVGPREC = 0
		sum_RANKLOSS = 0
		for i in range(num_labels):
			if label.__contains__(str(i)) and label[str(i)] == "1":
				#AVGPREC
				num_pos += 1
				i_vote = prediction[int(i)]
				LA = 0
				position = 1
				for j in range(len(prediction) - 1):
					if prediction[j] > i_vote:
						position += 1
						if label.__contains__(str(j)) and label[str(j)] == "1":
							LA += 1
				sum_AVGPREC += LA / position
				#RANKLOSS
				for j in range(num_labels):
					if (label.__contains__(str(j)) and label[str(j)] == "0") or label.__contains__(str(j)) == False:
						if prediction[int(i)] < prediction[int(j)]:
							sum_RANKLOSS += 1
			else:
				num_neg += 1
		if num_pos == 0:
			AVGPREC += 0
		else:
			AVGPREC += sum_AVGPREC / num_pos
		if num_pos * num_neg == 0:
			RANKLOSS += 0
		else:
			RANKLOSS += sum_RANKLOSS / (num_pos * num_neg)
	print(RP,IP,RN,IN)
	Accuracy_score = (num_datas - mistake) / num_datas
	AVGPREC = AVGPREC / num_datas
	RANKLOSS = RANKLOSS / num_datas
	P_hat = RP + IP
	N_hat = RN + IN
	P = RP + RN
	N = IP + IN
	L = P + N
	HamLoss = 1 - (RN + IP) / L
	if P_hat == 0:
		Precision = 0
	else:
		Precision = RP / P_hat
	if P == 0:
		Recall = 0
	else:
		Recall = RP / P
	if Precision + Recall == 0:
		F1 = 0
	else:
		F1 = 2 * Recall * Precision / (Recall + Precision)
	
	
	#the label with most vote correct
	#	most = 0
	#	most_i = -1
	#	for i in range(len(prediction) - 1):
	#		if prediction[i] > most:
	#			most = prediction[i]
	#			most_i = i
	#	if most < prediction[num_labels]:
	#		pred = 0
	#	else:
	#		pred = 1
	#	if label.__contains__(str(most_i)) == False:
	#		value = 0
	#	else:
	#		value = int(label[str(most_i)])
	#	if value != pred:
	#		mistake += 1
				
	print("Accuracy_score=%.4f\nHamLoss=%.4f\nPrecision=%.4f\nRecall=%.4f\nF1=%.4f\nAVGPREC=%.4f\nRANKLOSS=%.4f" % (Accuracy_score, HamLoss, Precision, Recall, F1, AVGPREC, RANKLOSS))
	return Accuracy_score, HamLoss, Precision, Recall, F1, AVGPREC, RANKLOSS
	

def compute_logp(data, i, j, posibilities, num_attributes, attr_type_flag, max_and_min, data_label):
	logp_1 = 0
	logp_0 = 0
	for k in range(num_attributes):
		if data.__contains__(str(k)) == False:
			value = "0"
		else:
			value = data[str(k)]
		if attr_type_flag[str(k)] == "0":
			#logp_1 += math.log(float(posibilities["%s=%s|%s_%s=1" %(str(k), value, str(i), str(j))]) * float(posibilities["%s_%s=1" % (str(i), str(j))]))
			#logp_0 += math.log(float(posibilities["%s=%s|%s_%s=0" %(str(k), value, str(i), str(j))]) * float(posibilities["%s_%s=0" % (str(i), str(j))]))
			logp_1 += math.log(float(posibilities["%s=%s|%s_%s=1" %(str(k), value, str(i), str(j))]))
			logp_0 += math.log(float(posibilities["%s=%s|%s_%s=0" %(str(k), value, str(i), str(j))]))
		else:
			max = float(max_and_min["%s_max" % str(k)])
			min = float(max_and_min["%s_min" % str(k)])
			one_piece = (max - min) / NUM_BINS
			type = NUM_BINS - 1
			for num in range(1, NUM_BINS):
				if float(value) < min + num * one_piece:
					type = num - 1
					break
			#logp_1 += math.log(float(posibilities["%s=%s|%s_%s=1" %(str(k), type, str(i), str(j))]) * float(posibilities["%s_%s=1" % (str(i), str(j))]))
			#logp_0 += math.log(float(posibilities["%s=%s|%s_%s=0" %(str(k), type, str(i), str(j))]) * float(posibilities["%s_%s=0" % (str(i), str(j))]))
			logp_1 += math.log(float(posibilities["%s=%s|%s_%s=1" %(str(k), type, str(i), str(j))]))
			logp_0 += math.log(float(posibilities["%s=%s|%s_%s=0" %(str(k), type, str(i), str(j))]))
			#print("1:",posibilities["%s=%s|%s_%s=1" %(str(k), type, str(i), str(j))],"2:",posibilities["%s_%s=1" % (str(i), str(j))],"3:",posibilities["%s=%s|%s_%s=0" %(str(k), type, str(i), str(j))],"4:",posibilities["%s_%s=0" % (str(i), str(j))])
			#mean_1 = float(mean_and_var["mean_%s|%s_%s=1" % (k, i, j)])
			#mean_0 = float(mean_and_var["mean_%s|%s_%s=0" % (k, i, j)])
			#var_1 = float(mean_and_var["var_%s|%s_%s=1" % (k, i, j)])
			#var_0 = float(mean_and_var["var_%s|%s_%s=0" % (k, i, j)])
			#logp_1 += math.log(1 / math.sqrt(2 * math.pi * var_1) * (math.e ** (-((float(value) - mean_1) ** 2)/(2 * var_1))))
			#logp_0 += math.log(1 / math.sqrt(2 * math.pi * var_0) * (math.e ** (-((float(value) - mean_0) ** 2)/(2 * var_0))))
	if float(posibilities["%s_%s=1" % (str(i), str(j))]) == 0:
		logp_1 += 0
	else:
		logp_1 += math.log(float(posibilities["%s_%s=1" % (str(i), str(j))]))
	if float(posibilities["%s_%s=0" % (str(i), str(j))]) == 0:
		logp_0 += 0
	else:
		logp_0 += math.log(float(posibilities["%s_%s=0" % (str(i), str(j))]))
	if logp_1 > logp_0:
		data_label["%s_%s" % (i, j)] = "1"
	elif logp_0 > logp_1:
		data_label["%s_%s" % (i, j)] = "0"
	else:
		rand = random.random()
		if rand < 0.5:
			data_label["%s_%s" % (i, j)] = "0"
		else:
			data_label["%s_%s" % (i, j)] = "1"
	return