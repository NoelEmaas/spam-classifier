import pandas as pd
import re
from nltk.tokenize import word_tokenize
from naive_bayes_spam_classifier import NaiveBayesSpamClassifier


def main ():
    # Read the data
    training_data = pd.read_csv('../data/TrainingData.csv', encoding='ISO-8859-1')
    testing_data = pd.read_csv('../data/TestData.csv', encoding='ISO-8859-1')

    # Cleaning and preprocessing of data
    training_data = preprocess_data(training_data)
    testing_data = preprocess_data(testing_data)

    # Create a NaiveBayesSpamClassifier object
    classifier = NaiveBayesSpamClassifier()

    # Train the classifier
    classifier.train(training_data)

    # Test the classifier
    classifier.test(testing_data)


def preprocess_data (data):
    data = data.drop_duplicates()
    data = data.dropna()

    def clean_message(message):
        message = message.lower()
        message = re.sub(r'[^\w\s]', '', message)
        words = word_tokenize(message)
        words = [word for word in words if word.isalpha()]
        cleaned_message = ' '.join(words)
        return cleaned_message

    data['message'] = data['message'].apply(clean_message)
    return data


if __name__ == "__main__":
    main()