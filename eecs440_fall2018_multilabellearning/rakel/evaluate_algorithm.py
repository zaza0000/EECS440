
def accuracy(prediction,lp_label_test):

    numerator = 0
    denominator = 0
    for key in prediction.keys():
        if prediction[key] == lp_label_test[str(key)]:
            # print(prediction[key])
            # print(lp_label_test[str(key)])
            numerator += 1
        denominator += 1
    acc = numerator/denominator
    return acc

def confusion_matrix(binary_prediction, testing_set_label):

    label_confusion_parameter = {}
    for key in binary_prediction[0].keys():
        # label_confusion_parameter[key] = {}
        tp,fp,fn,tn = 0,0,0,0
        for nExample in range(len(binary_prediction)):
            if binary_prediction[nExample][key] == 1 and testing_set_label[nExample][key] == '1':
                tp+=1
            if binary_prediction[nExample][key] == 1 and testing_set_label[nExample][key] == '0':
                fp+=1                
            if binary_prediction[nExample][key] == 0 and testing_set_label[nExample][key] == '1':
                fn+=1
            if binary_prediction[nExample][key] == 0 and testing_set_label[nExample][key] == '0':
                tn+=1     
        label_confusion_parameter[key] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn }

    return label_confusion_parameter

def cal_f1(label_confusion_parameter):

    label_f1 = {}
    label_precision = {}
    label_recall = {}
    for key in label_confusion_parameter.keys():
        tp, fp, fn, tn = label_confusion_parameter[key]['tp'], label_confusion_parameter[key]['fp'],label_confusion_parameter[key]['fn'],label_confusion_parameter[key]['tn']
        if tp == 0 and fp == 0 and fn == 0:
            label_f1[key] = 0
        else:
            label_f1[key] = 2*tp/(2*tp+fp+fn)
        if tp + fp == 0:
            label_precision[key] = 1
        else:
            label_precision[key] = tp/(tp+fp)
        if tp + fn == 0:
            label_recall[key] = 1
        else:
            label_recall[key] = tp/(tp+fn)
        # label_recall[key] = tp/(tp+fn)
    f1 = sum(label_f1.values()) / len(label_f1.values())
    precision = sum(label_precision.values()) / len(label_precision.values())
    recall = sum(label_recall.values()) / len(label_recall.values())

    return f1, precision, recall


def accuracy_matrix(label_confusion_parameter):
     
    acc_dict = {}
    for key in label_confusion_parameter.keys():
        tp, fp, fn, tn = label_confusion_parameter[key]['tp'], label_confusion_parameter[key]['fp'],label_confusion_parameter[key]['fn'],label_confusion_parameter[key]['tn']    
        acc_dict[key] = (tp+tn) / (tp+tn+fp+fn)

    return acc_dict