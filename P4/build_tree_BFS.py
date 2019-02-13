#!/usr/bin/env python
from mldata import *
from math import log
import numpy as np
from IG_v5 import *


# ====================================
# load_data(data_name) 
# input:            filename+dir
# output:           dataset
# Sep.14.2018
# description:
#   load data from files
# ====================================
def load_data(data_name):
    data = parse_c45(data_name)
    num_data = len(data)
    if(num_data == 0):
        print ("empty dataset")
        return None
    return ExampleSet(data)

# (not use)
# ====================================
# calc_entropy(num_of_data_inclass) 
# input:            number of data in each class
# output:           entropy
# Sep.14.2018
# description:
#   compute entropy H()
# ====================================
def calc_entropy(num_of_data_inclass):
    totalNum = sum(num_of_data_inclass)
    entropy = 0.0
    try: 
        for each in num_of_data_inclass:
            prob = 1.0 * each/totalNum
            entropy += -prob * log(prob ,2)
        return entropy
    except ValueError: #0log0
        return entropy

# (not use)
# ====================================
# class_entropy(dataset)
# input:            dataset
# output:           class entropy
# Sep.14.2018
# description:
#   compute class entropy H()
# ====================================
def class_entropy(dataset):
    num_of_data_inclass = []
    class1 = ExampleSet(attri for attri in data if attri[5] == 0)
    num_of_data_inclass.append(len(class1))
    class2 = ExampleSet(attri for attri in data if attri[5] == 1)
    num_of_data_inclass.append(len(class2))
    return calc_entropy(num_of_data_inclass)
    #return calc_entropy(num_of_data_inclass)

# (not use)
# ====================================
# attribute_entropy(attri_index,dataset)
# input:            attribute index, dataset
# output:           attribute entropy
# Sep.14.2018
# description:
#   compute attribute entropy H()
# ====================================
def attribute_entropy(attri_index,dataset):
    

    b = [9.0/16, 7.0/16]
    return calc_entropy(b)

# ====================================
# info_gain(dataset)
# input:            dataset
# output:           information gain
# Sep.26.2018
# description:
#   compute information gain
# ====================================
def info_gain_or_gain_ratio(attribute_index, dataset, use_gain_ratio = 0):

    try:
        data_array = np.array(dataset.to_float())
        none_row, none_column = np.where(data_array == None)
    except ValueError:
        return 0, None
    
    for x in none_column:
        if(x == attribute_index):
            index = np.where(none_column == attribute_index)
            data_array = np.delete(data_array,none_row[index],0)

    threshold = None;
    if (dataset.schema[attribute_index].type == 'CONTINUOUS'):
        data_arranged, value_ratio = continuous_data(data_array, attribute_index)
        threshold, hyx = find_minima_threshold(value_ratio)

    else: 
        data_arranged,value_ratio = discrete_data(data_array, attribute_index)
        hyx = calculate_hyx(value_ratio)
    ig = information_gain(data_array,hyx)

    #if(use_gain_ratio == 1):
    #    print("--------->",ig, value_ratio)
    #    g_ratio = gain_ratio(ig, value_ratio)
    #    ig = g_ratio

    #print(ig, threshold)
    return ig, threshold

# ====================================
# get_majority_class(dataset):
# input:            dataset
# output:           class label
# Sep.17.2018
# description:
#   majority class label
# ====================================
def get_majority_class(dataset):
    pos_data = ExampleSet(subset_data for subset_data in dataset if subset_data[-1] == 1)
    pos_data_num = len(pos_data)
    neg_data = ExampleSet(subset_data for subset_data in dataset if subset_data[-1] == 0)
    neg_data_num = len(neg_data)
    if(pos_data_num >= neg_data_num):
        return 1
    else:
        return 0

# ====================================
# get_majority_class(dataset):
# input:            attribute index， dataset
# output:           possible value in the dataset
# Sep.21.2018
# description:
#   the schema have all the value that may
# exsit, however, only some of them appears 
# in the dataset, this function is going to find
# out all possible (exsit) values
# ====================================
def get_possible_value(attri_index, dataset):
    possible_value = list()
    i = 0
    for value in dataset.schema[attri_index].values:
        if(len(ExampleSet(subset_data for subset_data in dataset if subset_data[attri_index] == value)) > 0):
            possible_value.append(value)
            i += 1
    return possible_value



# ====================================
# class TreeNode(object):
# Sep.21.2018
# description:
#   Tree node
# ====================================
class TreeNode(object):

    def __init__(self, attri_index=None, attri_value=None, parent=None, depth=0, is_continuous=False, is_pureNode=False, label=None):
        self._attri_index = attri_index
        self._attri_value = attri_value
        self.depth = depth
        self.childrenNum = 0
        self.__children = dict()
        self.__parent = parent
        self.is_continuous = is_continuous
        self.is_pureNode = is_pureNode
        self._label = label

    def get_children(self):
        return self.__children

    def get_attriIndex(self):
        return self._attri_index

    def get_attriValue(self):
        return self._attri_value

    def get_label(self):
        return self._label

    def set_attri_index(self, attri_index):
        self._attri_index = attri_index

    def set_attri_value(self, attri_value):
        self._attri_value = attri_value

    def set_parent(self, parent):
        self.parent = parent

    def set_label(self, label):
        self._label = label

    def get_parent(self):
        return self.parent

    def add_child(self, new_children):
        self.__children = new_children
        self.childrenNum = len(new_children)
        #new_children.set_parent(self)


# ====================================
# build_DecisionTree(object):
# Sep.21.2018
# description:
#   Build a decision tree
# ====================================
class build_DecisionTree(object):

    def __init__(self, max_depth=0, eps=0, trainning_dataset=None, use_gain_ratio=0, attri_indices = None):
        if(trainning_dataset == None):
            print ("No trainning data provided")
            return None

        self.max_depth = max_depth
        self.eps = eps
        self.attri_indices = attri_indices
        self.size = 1
        self.depth = 0
        self.use_gain_ratio = use_gain_ratio;
        self.root = self.buildTree(trainning_dataset)

    def get_root(self):
        return self.root

    def get_tree_depth(self):
        return self.depth

    def get_tree_size(self):
        return self.size

# ====================================
# get_best_attribute(self, dataset):
# Sep.21.2018
# description:
#   get the best attribute means get the 
# max IG and the related attribute
# ====================================

    def get_best_attribute(self, dataset):
        the_best_one = 0 # the best one have the max information gain
        featureList = self.attri_indices
        max_IG_attri = None
        max_IG_attri_value = 0
        max_IG = self.eps
        for i in featureList:
            IG, attri_value = info_gain_or_gain_ratio(i, dataset, self.use_gain_ratio)
            if(IG > max_IG):
                max_IG = IG
                max_IG_attri = i
                max_IG_attri_value = attri_value
        #print ("getBestAttri---->",max_IG,max_IG_attri)
        return max_IG_attri, max_IG_attri_value

# ====================================
# generate_children
# Sep.21.2018
# description:
#   use recursion to build a tree
# ====================================
    def generate_children(self, feature_index, feature_value, dataset, current_depth):
        if(current_depth > self.depth):   #update tree depth
            self.depth = current_depth

        new_children = dict()
        attri_type = dataset.schema[feature_index].type
        if(attri_type == "CONTINUOUS"):
            # new_children[0] = value not larger than feature_value
            subset_notlarger = ExampleSet(subset_data for subset_data in dataset if subset_data[feature_index] <= feature_value)
            next_feature_index, next_feature_value = self.get_best_attribute(subset_notlarger)
            child_0 = TreeNode(depth = current_depth)
            child_0.set_label(get_majority_class(subset_notlarger))
            if(next_feature_index == None or current_depth == self.max_depth):  # stop when there IG = 0 or reach the max depth
                child_0.is_pureNode = True
                new_children[0] = child_0
            else:
                child_0.set_attri_index(next_feature_index)
                if(dataset.schema[next_feature_index].type == "CONTINUOUS"):
                    child_0.is_continuous = True
                    child_0.set_attri_value(next_feature_value)
                child_0.add_child(self.generate_children(next_feature_index, next_feature_value, subset_notlarger, current_depth+1))
                new_children[0] = child_0

            # new_children[1] = value larger than feature_value
            subset_larger = ExampleSet(subset_data for subset_data in dataset if subset_data[feature_index] > feature_value)
            next_feature_index, next_feature_value = self.get_best_attribute(subset_larger)
            child_1 = TreeNode(depth = current_depth)
            child_1.set_label(get_majority_class(subset_larger))
            if(next_feature_index == None or current_depth == self.max_depth):
                child_1.is_pureNode = True
                new_children[1] = child_1
            else:
                child_1.set_attri_index(next_feature_index)
                if(dataset.schema[next_feature_index].type == "CONTINUOUS"):
                    child_1.is_continuous = True
                    child_1.set_attri_value(next_feature_value)
                child_1.add_child(self.generate_children(next_feature_index, next_feature_value, subset_larger, current_depth+1))
                new_children[1] = child_1

        elif(attri_type == "NOMINAL"):
            possible_value = get_possible_value(feature_index, dataset)
            for i in possible_value:
                subset = ExampleSet(subset_data for subset_data in dataset if subset_data[feature_index] == i)
                next_feature_index, next_feature_value = self.get_best_attribute(subset)
                new_child = TreeNode(depth = current_depth)
                new_child.set_label(get_majority_class(subset))
                if(next_feature_index == None or current_depth == self.max_depth):
                    new_child.is_pureNode = True
                    new_children[i] = new_child
                else:
                    new_child.set_attri_index(next_feature_index)
                    if(dataset.schema[next_feature_index].type == "CONTINUOUS"):
                        new_child.is_continuous = True
                        new_child.set_attri_value(next_feature_value)
                    new_child.add_child(self.generate_children(next_feature_index, next_feature_value, subset, current_depth+1))
                    new_children[i] = new_child

        else: # BINARY
            possible_value = {0,1}
            for i in possible_value:
                subset = ExampleSet(subset_data for subset_data in dataset if subset_data[feature_index] == i)
                next_feature_index, next_feature_value = self.get_best_attribute(subset)
                new_child = TreeNode(depth = current_depth)
                new_child.set_label(get_majority_class(subset))
                if(next_feature_index == None or current_depth == self.max_depth):
                    new_child.is_pureNode = True
                    new_children[i] = new_child
                else:
                    new_child.set_attri_index(next_feature_index)
                    if(dataset.schema[next_feature_index].type == "CONTINUOUS"):
                        new_child.is_continuous = True
                        new_child.set_attri_value(next_feature_value)
                    new_child.add_child(self.generate_children(next_feature_index, next_feature_value, subset, current_depth+1))
                    new_children[i] = new_child

        self.size += len(new_children)  # update tree size

        return new_children


    def buildTree(self, dataset):
        root = TreeNode(depth=0)
        feature_index, feature_value = self.get_best_attribute(dataset)
        root.set_label(get_majority_class(dataset))
        if(feature_index == None):
            print ("Can not do the first split")
            root.is_pureNode = True
            return root
        root.set_attri_index(feature_index)
        if(dataset.schema[feature_index].type == "CONTINUOUS"):
            root.is_continuous = True
            root.set_attri_value(feature_value)
        root.add_child(self.generate_children(feature_index, feature_value, dataset, root.depth+1))

        return root

    def classify_data2(self, current_node, example):
        predicted_label = None
        children = current_node.get_children()
        current_attri_index = current_node.get_attriIndex()


        if(example[current_attri_index] == None):    # example does not have this atrribute, return the majority class label
            return current_node.get_label()

        if(current_node.is_continuous == True):  # for continuous variable, compare the value first.
            if(example[current_attri_index] <= current_node.get_attriValue()):
                if(children[0].is_pureNode == True):   # next node is pure node, return the majority class label
                    return children[0].get_label()
                else:
                    predicted_label = self.classify_data2(children[0], example)
            else:
                if(children[1].is_pureNode == True):
                    return children[1].get_label()
                else:
                    predicted_label = self.classify_data2(children[1], example)
        else:
            try:    # if the tree does not have this value, return the majority class
                next_node = children[example[current_attri_index]]
                if(next_node.is_pureNode == True):   # next node is pure node, return the majority class label
                    return next_node.get_label()
                else:
                    predicted_label = self.classify_data2(next_node, example)
            except KeyError:
                #print ("------> KeyError")
                return current_node.get_label()

        return predicted_label


    def classify_data(self, example):
        #1. is pure node?
        #   Y: return label, N:go to 2.
        #2. attri_index:
        #   continuous:  use attri_value to find relavant child,  others: use subscript to find value,  None: majority class
        predicted_label = None
        if(self.root.is_pureNode == True):
            return self.root.get_label()
        else:
            predicted_label = self.classify_data2(self.root, example)
        return predicted_label

    def classify_dataset(self, dataset):
        data_size = len(dataset)
        if(data_size == 0):
            print ("No example provided")
            return None
        correct_num = 0
        for ex in dataset:     
            predict_label = self.classify_data(ex)
            if(predict_label == ex[-1]):
                correct_num += 1

        return correct_num/data_size


# ====================================
# main function
# ====================================
if __name__ == '__main__':

    print ("step1: load data")
    training_data = load_data("example")  #load data
    testing_data = training_data
    print ("...")

    print ("step2: build decision tree")
    dtree = build_DecisionTree(max_depth=50, eps=1.0e-10, trainning_dataset=training_data, use_gain_ratio=1) #build decision tree
    print ("...")

    print ("step3: classify data")
    print ("...")
    print ("accuary: ",dtree.classify_dataset(testing_data))
    print ("-------------------------------")
    print("tree size: ", dtree.get_tree_size())
    print("tree depth: ", dtree.get_tree_depth())
    print("first feature index: ", dtree.get_root().get_attriIndex())


    #print dtree.root.get_children()[0].get_label()
    #print (ExampleSet(subset_data for subset_data in data if subset_data[2] == "Monday"))

    #a = ExampleSet(subset_data for subset_data in data if subset_data[4] == None)
    #print a
    #a = 2.5
    #if(a == 1.5):
    #    print tests[a]




