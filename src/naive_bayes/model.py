import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)

class NaiveBayesSpamClassifier:
    def __init__ (self):
        self.vocabulary = set() # storage for unique words
        self.total_number_of_words = 0 # counter for total number of words
        self.spam_word_probabilities = defaultdict() # storage for probability of each word in spam messages
        self.ham_word_probabilities = defaultdict() # storage for probability of each word in ham messages


    # Train the classifier
    def train (self, training_data):
        print('Training the classifier...')

        spam_word_counts = Counter() # storage for count of each word in spam messages
        ham_word_counts = Counter() # storage for count of each word in ham messages
        spam_words = set()  # storage for unique words in spam messages
        ham_words = set() # storage for unique words in ham messages

        # Get the count of each of word as spam or ham
        # Get the list of unique words
        for _, row in training_data.iterrows():
            message = row['message']
            label = row['label']

            words = word_tokenize(message) # extract the words in the message
            self.total_number_of_words += len(words) # update the total number of words
            self.vocabulary.update(words) # update the vocabulary

            # Update the count of each word as spam or ham
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

        # Iterate through every message
        for _, row in test_data.iterrows():
            probability_spam = 1 # store probability of message being spam
            probability_ham = 1 # store probability of message being ham

            message = row['message']
            words = word_tokenize(message)

            # Compute the smoothing factor for words that does not exist in the training data but exist in the test data
            smoothing_factor = 1 / (self.total_number_of_words + len(self.vocabulary) + 1)

            # Compute the total probability of the message as spam or han
            for word in words:
                probability_spam *= self.spam_word_probabilities.get(word, smoothing_factor)
                probability_ham *= self.ham_word_probabilities.get(word, smoothing_factor)

            # Compare the probability of the message as spam or ham
            test_result.append('spam' if probability_spam > probability_ham else 'ham')
        
        print('Prediction complete!\n')
        return test_result
    
    
    # Cleaning of data
    def preprocess (self, data):
        stop_words = stopwords.words('english') # get the list of stop words

        # Function to clean the message
        def clean_message(message):
            message = message.lower() # convert to lowercase
            message = re.sub(r'[^\w\s]', '', message) # remove punctuation
            words = word_tokenize(message) # extract the words in the message
            words = [word for word in words if (word not in stop_words) and word.isalpha()] # remove stop words and non-alphabetic words
            cleaned_message = ' '.join(words) # join the words back into a message
            return cleaned_message

        # Clean the message from the data
        data['message'] = data['message'].apply(clean_message)
        return data
