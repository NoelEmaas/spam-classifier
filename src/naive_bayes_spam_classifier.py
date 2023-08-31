import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter


class NaiveBayesSpamClassifier:
    def __init__ (self):
        self.vocabulary = set()
        self.total_number_of_words = 0
        self.spam_word_probabilities = defaultdict()
        self.ham_word_probabilities = defaultdict()


    # Train the classifier
    def train (self, training_data):
        print('Training the classifier...')

        spam_word_counts = Counter()
        ham_word_counts = Counter()
        spam_words = set()
        ham_words = set()

        for _, row in training_data.iterrows():
            message = row['message']
            label = row['label']

            words = word_tokenize(message)
            self.total_number_of_words += len(words)
            self.vocabulary.update(words)

            if label == 'spam':
                spam_word_counts.update(words)
                spam_words.update(words)
            else:
                ham_word_counts.update(words)
                ham_words.update(words)
        
        # Get the probability of each word in spam and ham messages
        for word in self.vocabulary:
            self.spam_word_probabilities[word] = (spam_word_counts[word] + 1) / (len(spam_words) + len(self.vocabulary))
            self.ham_word_probabilities[word] = (ham_word_counts[word] + 1) / (len(ham_words) + len(self.vocabulary))

        print("Training complete!\n")


    # Predict the label of a message based on trainging data
    def test (self, test_data):
        print('Predicting labels for test data...')

        test_result = []

        for _, row in test_data.iterrows():
            probability_spam = 1
            probability_ham = 1

            message = row['message']
            words = word_tokenize(message)

            smoothing_factor = 1 / (self.total_number_of_words + len(self.vocabulary) + 1)

            for word in words:
                probability_spam *= self.spam_word_probabilities.get(word, smoothing_factor)
                probability_ham *= self.ham_word_probabilities.get(word, smoothing_factor)

            test_result.append('spam' if probability_spam > probability_ham else 'ham')
        
        print('Prediction complete!\n')

        self._store_result_in_csv(test_result)
        
            
    # Store result in new csv file which contains the predicted label, and message
    def _store_result_in_csv (self, test_result):
        orig_test_data = pd.read_csv('../data/TestData.csv', encoding='ISO-8859-1')
        orig_test_data['label'] = test_result

        # Reorder columns to have the label column as the first column
        column_order = ['label'] + [col for col in orig_test_data.columns if col != 'label']
        orig_test_data = orig_test_data[column_order]

        # Save the result in a csv file
        test_result_file_name = '../data/EmaasResultData.csv'
        orig_test_data.to_csv(test_result_file_name, index = False) 

        print('Successfully saved predicted data!')
        