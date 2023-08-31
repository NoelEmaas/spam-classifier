import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from naive_bayes_spam_classifier import NaiveBayesSpamClassifier

# download stop words
nltk.download('stopwords', quiet = True)

def main ():
    # Read the data
    training_data = pd.read_csv('../data/TrainingData.csv', encoding='ISO-8859-1')
    testing_data = pd.read_csv('../data/TestData.csv', encoding='ISO-8859-1')

    # Cleaning and preprocessing of data
    training_data = preprocess_data(training_data)
    training_data = training_data.drop_duplicates()
    training_data = training_data.dropna()
    testing_data = preprocess_data(testing_data)

    # Create a NaiveBayesSpamClassifier object
    classifier = NaiveBayesSpamClassifier()

    # Train the classifier
    classifier.train(training_data)

    # Test the classifier
    classifier.test(testing_data)


#Preprocess and clean the data
def preprocess_data (data):
    stop_words = stopwords.words('english')

    def clean_message(message):
        message = message.lower()
        message = re.sub(r'[^\w\s]', '', message)
        words = word_tokenize(message)
        words = [word for word in words if (word not in stop_words) and word.isalpha()]
        cleaned_message = ' '.join(words)
        return cleaned_message

    data['message'] = data['message'].apply(clean_message)
    return data


if __name__ == "__main__":
    main()