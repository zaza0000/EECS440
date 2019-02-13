import label_parser

def data_processing(filename, datatype):
	attr_type_flag = {}		# store the type('1' for numeric and '0' for discrete) of all possible attributes. {attr_num: '1', ...}
	full_labels = {}		# store all the possible labels. {label_number: label_name, ...}
	data_value_list = []	# store attributes of all data. [{attr_num: attr_value, ...}, ...]
	data_label_list = []	# store labels of all data. [{label_num: label_value, ...}, ...]
							# attr_num and label_num are not successive, for instance, attr_num is 0 to 500, label_num is 0 to 5.
							# all of the keys and values are string.
	if datatype == 0:
		file = open("./Datasets/" + filename + "/" + filename + ".arff")
	elif datatype == 1:
		file = open("./Datasets/" + filename + "/" + filename + "-train.arff")
	elif datatype == 2:
		file = open("./Datasets/" + filename + "/" + filename + "-test.arff")
	else:
		raise ValueError("datatype should be 0(for full dataset) or 1(for training set) or 2(for testing set)")
	lines = file.readlines()
	file.close()
	labels = label_parser.LabelParser("./Datasets/" + filename + "/" + filename + ".xml")
	data_flag = 0
	num = 0
	full_labels_flag = 0
	for line in lines:
		if line.startswith("@attribute") == True:
			line = line[11:-1]
			partition = line.partition(" ")
			name = partition[0]
			type = partition[2]
			if labels.count(name) == 0:
				if type == "numeric":
					attr_type_flag[str(num)] = "1"
				elif type == "{0,1}":
					attr_type_flag[str(num)] = "0"
			else:
				if full_labels_flag == 0:
					first_label_num = num
					full_labels_flag = 1
				full_labels[str(num - first_label_num)] = name
			num += 1
		elif line.startswith("@data"):
			data_flag = 1
		elif data_flag == 1 and line != "\n":
			if line.startswith("{"):
				data_value_dic = {}
				data_label_dic = {}
				line = line[1:-2]
				while "," in line:
					line_part = line.partition(",")
					data = line_part[0]
					line = line_part[2]
					data_part = data.partition(" ")
					if full_labels.__contains__(str(int(data_part[0]) - first_label_num)) == False:
						data_value_dic[data_part[0]] = data_part[2]
					else:
						data_label_dic[str(int(data_part[0]) - first_label_num)] = data_part[2]
				data_part = line.partition(" ")
				if full_labels.__contains__(str(int(data_part[0]) - first_label_num)) == False:
					data_value_dic[data_part[0]] = data_part[2]
				else:
					data_label_dic[str(int(data_part[0]) - first_label_num)] = data_part[2]
				data_value_list.append(data_value_dic)
				data_label_list.append(data_label_dic)
			else:
				data_value_dic = {}
				data_label_dic = {}
				line = line[0:-1]
				num = 0
				while "," in line:
					line_part = line.partition(",")
					data = line_part[0]
					line = line_part[2]
					if full_labels.__contains__(str(num - first_label_num)) == False:
						data_value_dic[str(num)] = data
					else:
						data_label_dic[str(num - first_label_num)] = data
					num += 1
				if full_labels.__contains__(str(num - first_label_num)) == False:
					data_value_dic[str(num)] = line
				else:
					data_label_dic[str(num - first_label_num)] = line
				data_value_list.append(data_value_dic)
				data_label_list.append(data_label_dic)
	return attr_type_flag, full_labels, data_value_list, data_label_list
			
if __name__ == "__main__":
	data_processing("scene", datatype = 2)