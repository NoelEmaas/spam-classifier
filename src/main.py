import pandas as pd
from naive_bayes.model import NaiveBayesSpamClassifier
from naive_bayes.metrics import compute_confusion_values, compute_precision, compute_recall

def main ():
    # Read the data
    training_data = pd.read_csv('../data/TrainingData.csv', encoding='ISO-8859-1')
    testing_data = pd.read_csv('../data/TestData.csv', encoding='ISO-8859-1')

    # Create a NaiveBayesSpamClassifier object
    classifier = NaiveBayesSpamClassifier()

    # Cleaning of data
    training_data = classifier.preprocess(training_data)
    testing_data = classifier.preprocess(testing_data)

    # Train classifier
    classifier.train(training_data)

    # Test the classifier and store result
    prediction_result = classifier.test(testing_data)

    # Save prediction result to csv
    result_file_path = '../data/EmaasResultData.csv'
    testing_data.insert(0, 'label', prediction_result)
    testing_data.to_csv(result_file_path, index = False)
    print('Prediction result stored in EmaasResultData.csv file!')

    # store result data and labeled data path
    result_data_path = '../data/EmaasResultData.csv'
    labeled_data_path = '../data/LabeledTestData.csv'
    
    # read result data and label data using pandas
    resultDataFrame = pd.read_csv(result_data_path, encoding='ISO-8859-1')
    labeledDataFrame = pd.read_csv(labeled_data_path, encoding='ISO-8859-1')
    
    # compute the confusion values
    TP, TN, FP, FN = compute_confusion_values(resultDataFrame, labeledDataFrame)
    
    # compute for the precision and recall
    precision = compute_precision(TP, FP)
    recall = compute_recall(TP, FN)
   
    # print result
    print("Confusion values:")
    print("Number of TP:", TP)
    print("Number of TN:", TN)
    print("Number of FP:", FP)
    print("Number of FN:", FN)
    print("Precision value:", round(precision, 2))
    print("Recall value:", round(recall, 2)) 

    
if __name__ == "__main__":
    main()
