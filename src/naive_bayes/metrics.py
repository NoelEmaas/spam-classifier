import pandas as pd
import os
import sys


def compute_confusion_values (resultData, labeledData):
    # check first if the number of rows of result and label data are the same
    len_result_data = len(resultData)
    len_labeled_data = len(labeledData)
    if (len_result_data != len_labeled_data):
        print('Error: number of rows of result and label data is not the same!')
        print('result data rows:', len_result_data, ', labeled data rows:' , len_labeled_data)
        sys.exit(1)

    # create iter rows variable iterate through data
    iter_result_data = resultData.iterrows()
    iter_labeled_data = labeledData.iterrows()

    result_data_confusion_values = []
    tp, tn, fp, fn = [0, 0, 0, 0]
   
    # compute for confusion values
    for (_, row1), (_, row2) in zip(iter_result_data, iter_labeled_data):
        # true positive = spam message classified as spam
        if (row1['label'] == 'spam' and row2['label'] == 'spam'):
            result_data_confusion_values.append('TP')
            tp += 1
        # true negative = ham message classified as ham
        elif (row1['label'] == 'ham' and row2['label'] == 'ham'):
            result_data_confusion_values.append('TN')
            tn += 1
        # false positive = ham message misclassified as spam
        elif (row1['label'] == 'spam' and row2['label'] == 'ham'):
            result_data_confusion_values.append('FP')
            fp += 1
        # false negative = spam message misclassified as ham
        elif (row1['label'] == 'ham' and row2['label'] == 'spam'):
            result_data_confusion_values.append('FN')
            fn += 1

    # Store result in csv file
    curr_dir = os.getcwd()
    result_file_path = curr_dir + '/../data/EmaasPrecisionRecall.csv'
    resultData.insert(0, 'measure', result_data_confusion_values)
    resultData.insert(1, 'correct_label', labeledData['label'].tolist())
    resultData.to_csv(result_file_path, index = False)
    print('Precision and Recall result stored in EmaasPrecisionRecall.csv file!\n')
    
    return [tp, tn, fp, fn]


def compute_precision (TP, FP):
    return (TP / (TP + FP)) if (TP + FP != 0) else 0


def compute_recall (TP, FN):
    return (TP / (TP + FN)) if (TP + FN != 0) else 0



