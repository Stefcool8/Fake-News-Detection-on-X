import re

import contractions
import emoji
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from abbreviations import abbreviations

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Preprocess:
    def __init__(self, sub_words_path=None):
        self.stop_words = set(stopwords.words('english'))
        # Create an instance of WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()
        # POS tag mapping dictionary
        self.wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        self.sub_words_dict = self.get_sub_words_dictionary(sub_words_path) if sub_words_path else {}

    def clean_text(self, text):
        # Lowercase the text
        text = text.lower()
        # Fix encoding issues
        text = text.encode('latin1').decode('utf-8', errors='ignore')
        # Remove newlines
        text = text.replace('\n', ' ')
        # Removing URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Removing HTML tags
        text = BeautifulSoup(text, 'lxml').text
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Split and remove # sign, retain the hashtag words
        text = re.sub(r'#(\w+)', r' \1 ', text)
        # Remove extra spaces
        text = re.sub(' +', ' ', text)
        # Remove leading and trailing spaces
        text = text.strip()
        # Removing emojis
        text = self.convert_emojis_to_words(text)

        # Advanced preprocessing techniques
        lemmatized_words = self.advanced_preprocessing(text)

        return lemmatized_words

    def advanced_preprocessing(self, text):
        # Convert abbreviations to words
        text = self.convert_chat_words(text)
        # Remove contractions
        text = contractions.fix(text)
        # Remove punctuation and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Tokenize the text
        words = word_tokenize(text)
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        # Lemmatize the text
        lemmatized_words = self.lemmatize_text(words)

        return lemmatized_words

    def replace_missing_words(self, words):
        expanded_words = []
        has_replacements = False

        for word in words:
            if word in self.sub_words_dict:
                replacement_words = self.sub_words_dict[word]
                expanded_words.extend(replacement_words)
                has_replacements = True
            else:
                expanded_words.append(word)

        if has_replacements:
            expanded_text = " ".join(expanded_words)
            return self.advanced_preprocessing(expanded_text)
        else:
            return words

    def lemmatize_text(self, words):
        # Get the POS tags for the words
        pos_tags = nltk.pos_tag(words)

        # Perform Lemmatization
        lemmatized_words = []

        for word, tag in pos_tags:
            # Map the POS tag to WordNet POS tag
            pos = self.wordnet_map.get(tag[0].upper(), wordnet.NOUN)
            # Lemmatize the word with the appropriate POS tag
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=pos)
            # Add the lemmatized word to the list
            lemmatized_words.append(lemmatized_word)

        return lemmatized_words

    @staticmethod
    def convert_chat_words(text):
        words = text.split()
        converted_words = []
        for word in words:
            if word.lower() in abbreviations:
                converted_words.append(abbreviations[word.lower()])
            else:
                converted_words.append(word)
        converted_text = " ".join(converted_words)
        return converted_text

    # Function to convert emojis to words using emoji library mapping
    @staticmethod
    def convert_emojis_to_words(text):
        converted_text = emoji.demojize(text)
        return converted_text

    def filter_tweets(self, tweets, labels, min_length=1):
        filtered_tweets = []
        filtered_labels = []

        for tweet, label in zip(tweets, labels):
            cleaned_tweet = self.clean_text(tweet)

            # Replace missing words
            # cleaned_tweet = self.replace_missing_words(cleaned_tweet)

            if len(cleaned_tweet) >= min_length:
                filtered_tweets.append(cleaned_tweet)
                filtered_labels.append(label)
        return filtered_tweets, filtered_labels

    @staticmethod
    def get_sub_words_dictionary(dict_path):
        sub_words_dict = {}
        with open(dict_path, 'r') as f:
            for line in f:
                # First word is the key, the remaining words are the value
                words = line.split()
                value = [word for word in words[1:]]
                sub_words_dict[words[0]] = value
        return sub_words_dict
