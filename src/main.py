import pandas as pd
from naive_bayes_spam_classifier import NaiveBayesSpamClassifier

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
    print('Result saved in EmaasResultData.csv file!')


if __name__ == "__main__":
    main()
