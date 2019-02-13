import xml.sax
import numpy as np

############################################
# class Label:
#	A xml reader class. Called in LabelParser()
############################################
class Label(xml.sax.ContentHandler):
	def __init__(self):
		self.labels=[]
		
	def startElement(self, tag, attributes):
		if(tag == "label"):
			self.labels.append(attributes["name"])

	def getLabel():
		return self.labels

############################################
# LabelParser():
#	Using xml reader to do the label parsing
#	to get all the label names from xml file.
############################################		
def LabelParser(filepath):

	#XMLReader
	parser = xml.sax.make_parser()
	xmlreader = Label()
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)
	parser.setContentHandler(xmlreader)
	try:
		parser.parse(filepath)
	except:
		print("xml file not found")
		return None
	labels = xmlreader.labels
	return labels

############################################
# class Exampleset:
#	A class storing the pre-processed data 
#	and some parameters
############################################
class Exampleset():

	def __init__(self, dataset_name = None):
		if(dataset_name == None):
			print("no file input")
			return None
		self.dataset_name = dataset_name
		self.dataset = None		# data features + labels
		self.attri_names = None
		self.attri_types = None
		self.all_possi_labels = None
		self.num_of_labels = 0
		self.Load_data()

	############################################
	# Data_parser():
	#	load data from a given filepath
	############################################
	def Data_parser(self, filepath):
		try:
			file = open(filepath)
		except:
			print("arff file not found")
			return None
		lines = file.readlines()
		file.close()
		attri_names = []
		attri_types = []
		dataset = list()
		data_starts_flag = 0
		count = 0
		for line in lines:
			if(data_starts_flag == 1 and line != "\n"):
				data_x_line = list()
				if(line.startswith("{") == True):
					for i in range(count):
						data_x_line.append(0)
					line = line[1: len(line)-2]
					index = 0
					while "," in line:
						partition = line.partition(",")
						data = partition[0]
						partition2 = data.partition(" ")
						temp = int(partition2[0])
						line = partition[2]
						data_x_line[temp] = 1
					partition = line.partition(",")
					data = partition[0]
					partition2 = data.partition(" ")
					temp = int(partition2[0])
					data_x_line[temp] = 1
					dataset.append(data_x_line)
				else:
					while "," in line:
						partition = line.partition(",")
						data = float(partition[0])
						line = partition[2]
						data_x_line.append(data)
					data_x_line.append(float(line[0]))
					dataset.append(data_x_line)

			elif(line.startswith("@attribute") == True):
				count = count + 1
				temp = line.partition(" ")
				partition = temp[2].partition(" ")
				attri_name = partition[0]
				attri_type = partition[2]
				if(self.all_possi_labels.count(attri_name) == 0):
					attri_names.append(attri_name)
					attri_types.append(attri_type[0:-1])
			elif(line.startswith("@data") == True):
				data_starts_flag = 1

		return dataset, attri_names, attri_types

############################################
# Load_data():
#	load files by a given datasets name
############################################
	def Load_data(self):
		label_file_path = "./Datasets/" + self.dataset_name + "/" + self.dataset_name + ".xml"
		labels = LabelParser(label_file_path)
		if(labels == None):
			return None
		self.all_possi_labels = labels
		self.num_of_labels = len(self.all_possi_labels)
		data_file_path = "./Datasets/" + self.dataset_name + "/" + self.dataset_name + ".arff"
		self.dataset, self.attri_names, self.attri_types = self.Data_parser(data_file_path)





	