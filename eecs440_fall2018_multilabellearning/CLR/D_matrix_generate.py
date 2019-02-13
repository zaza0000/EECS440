# store Djk in Djk_label_list, [{'0_2': '1', '0_3': '1'}, ...], 0_2 is an example of j_k
# store Djv in Djv_label_list, [{'0_v': '0', '1_v': '1', ...}, ...]
def D_matrix_generate(full_labels, data_label_list):
	Djk_label_list = []
	Djv_label_list = []
	length = len(full_labels)
	for label in data_label_list:
		Djk_label = {}
		Djv_label = {}
		for j in range(length - 1):
			if label.__contains__(str(j)) and label[str(j)] == "1":
				Djv_label["%s_v" % str(j)] = "1"
			else:
				Djv_label["%s_v" % str(j)] = "0"
			for k in range(j+1, length):
				if label.__contains__(str(j)) and label.__contains__(str(k)):
					if label[str(j)] == "1" and label[str(k)] == "0":
						Djk_label["%s_%s" % (str(j), str(k))] = "1"
					elif label[str(j)] == "0" and label[str(k)] == "1":
						Djk_label["%s_%s" % (str(j), str(k))] = "0"
				elif label.__contains__(str(j)) and not label.__contains__(str(k)):
					if label[str(j)] == "1":
						Djk_label["%s_%s" % (str(j), str(k))] = "1"
				elif not label.__contains__(str(j)) and label.__contains__(str(k)):
					if label[str(k)] == "1":
						Djk_label["%s_%s" % (str(j), str(k))] = "0"
		if label.__contains__(str(length - 1)) and label[str(length - 1)] == "1":
			Djv_label["%s_v" % str(length - 1)] = "1"
		else:
			Djv_label["%s_v" % str(length - 1)] = "0"
		Djk_label_list.append(Djk_label)
		Djv_label_list.append(Djv_label)
	return Djk_label_list, Djv_label_list