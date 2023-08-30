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
        print('Classification Result:')

        for idx, row in test_data.iterrows():
            probability_spam = 1
            probability_ham = 1

            message = row['message']
            words = word_tokenize(message)

            smoothing_factor = 1 / (self.total_number_of_words + len(self.vocabulary) + 1)

            for word in words:
                probability_spam *= self.spam_word_probabilities.get(word, smoothing_factor)
                probability_ham *= self.ham_word_probabilities.get(word, smoothing_factor)

            print("[SPAM] " + message) if probability_spam > probability_ham else print("[HAM] " + message)
            
            if idx == 10:
                break