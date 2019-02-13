from load_data  import *
import data_processing
from constants import *
from Naive_Bayes_reborn import *
import cross_validation 
from evaluate_algorithm import *


def main():

    ave_acc = []
    ave_f1 = []
    ave_hemiloss = []
    ave_precision = []
    ave_recall = []
    attr_type_flag, full_labels, data_value_list, data_label_list = data_processing.data_processing(DATASET_NAME, 0)
    for i in range(len(attr_type_flag)):
        for j in range(len(data_value_list)):
            if data_value_list[j].__contains__(str(i))==False:
                data_value_list[j][str(i)]="0"
    for i in range(len(full_labels)):
        for j in range(len(data_label_list)):
            if data_label_list[j].__contains__(str(i))==False:
                data_label_list[j][str(i)]="0"
    datasets_value, datasets_label = cross_validation.construct_cv_folds(N_FOLDS, data_value_list, data_label_list)

    for i in range(N_FOLDS):
        training_set_value = []
        training_set_label = []
        for j in range(N_FOLDS):
            if i != j:
                training_set_value = training_set_value + datasets_value[j]
                training_set_label = training_set_label + datasets_label[j]
        testing_set_value = datasets_value[i]
        testing_set_label = datasets_label[i]

        #-------------------- train model -------------------
        # 1. Transform Mutilabel into powerset 
        lp_label, random_k_labels_str, M , numberD = label_powerset(training_set_label, k, 0, 1  )
        # 2. Train model
        conditional_prob_dict, threshold_dict = train_model(attr_type_flag, data_value_list ,training_set_value, numberD, lp_label, m_estimate) 
        print("-------- Finish Training --------")

        #-------------------- test  model -------------------
        # 1. convert continuous value into discrete
        continuous_to_discrete(testing_set_value, attr_type_flag, threshold_dict)
        # 2. test model -- predict
        prediction = test_model(testing_set_value , testing_set_label , conditional_prob_dict) 
        # 3. compare prediction and target
        lp_label_test, _, _, num_test = label_powerset(testing_set_label, k, random_k_labels_str, 2 )  
        # 4. evaluate methods
        # -- 4.1 convert to binary
        binary_prediction = convert_to_binary(prediction, k)
        # -- 4.2 confusion matrix
        label_confusion_parameter = confusion_matrix(binary_prediction, testing_set_label)
        # -- 4.3 Evaluation Parameters:
        f1, precision, recall = cal_f1(label_confusion_parameter)
        acc = accuracy(prediction,lp_label_test)
        acc_2 = accuracy_matrix(label_confusion_parameter)
        hemiloss = sum(acc_2.values())/k
        # 5. Sum up accuracy in each iteration
        ave_acc.append(acc)
        ave_f1.append(f1)
        ave_hemiloss.append(hemiloss)
        ave_precision.append(precision)
        ave_recall.append(recall)
        print("-------- Finish Testing --------")
    print("Dataset is :     ", DATASET_NAME)
    print("k =  ", k)
    print("accuracy---------", ave_acc)
    print("average accuracy-",sum(ave_acc)/N_FOLDS)
    print("hemiloss---------", [1-x for x in ave_f1])
    print("average hemiloss-", 1-sum(ave_f1)/N_FOLDS)
    print("precision--------", ave_precision)
    print("average precision", sum(ave_precision)/N_FOLDS)
    print("recall-----------", ave_recall)
    print("average recall---", sum(ave_recall)/N_FOLDS)
    print("f1---------------", ave_f1)
    print("average f1-------", sum(ave_f1)/N_FOLDS)



if __name__ == "__main__":
    main()