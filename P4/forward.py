from mldata import *
from math import log2
import numpy as np
from Naive_Bayes_reborn import *

def main():
    data_read = parse_c45("volcanoes")
    maxfeature = 5
    chosen_features = []
    save_continuous = []

    # a = np.array([[1,2,3,4,5],[6,7,8,9,0]])
    # print(np.where(a[:,2] == 8))
    # # print(a[-1])
    # # print(min(range(1,5)),max(range(1,5)))

    data_newarray = np.array(data_read.to_float())
    columns = np.shape(data_newarray)[1]
    rows = np.shape(data_newarray)[0]
    data_list = splitDataset(data_newarray,rows)
    

    #get max and min array
    new_data = data_newarray[:, 1:-1]
    maxminarray = np.zeros([2,len(new_data[0])])
    new_data = data_newarray[:, 1:-1] # only for naive bayes
    for i in range(1, columns-1):
        if data_read.schema.features[i].type == 'CONTINUOUS':
            maxminarray[0, i-1] = np.min(new_data[:,i-1])
            maxminarray[1, i-1] = np.max(new_data[:,i-1])
            save_continuous.append(i)


    saveTrainSet,saveValidationSet,saveValueRatio  = features_prob(maxminarray,save_continuous,data_list)     # train model     

    for i in range(5):
        unchosen_features = makeUnchosenColumns(columns, chosen_features)
        eRate = 0
        saveERate = []
        for attempFeature in unchosen_features:
            for nValidation in range(3):
                validaitonSetErrorRate = cvErrorRate(nValidation, saveValidationSet, saveValueRatio, chosen_features, attempFeature)
                eRate += validaitonSetErrorRate
            eRate = eRate/3
            saveERate.append([attempFeature,eRate])
        lowestErrorRate = np.min(saveERate[:,1])
        print("chosen features are :", chosen_features )
        print("error rate is :", lowestErrorRate )
        bestIndex = np.min(np.where(saveERate[:,1] == lowestErrorRate))
        chosen_columns.append(bestIndex) # update the chosen columns





if __name__ == '__main__':
    main()           
        
