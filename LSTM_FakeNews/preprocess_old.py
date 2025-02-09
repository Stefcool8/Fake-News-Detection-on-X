# import pandas as pd
# from html import unescape
# from sklearn.model_selection import train_test_split
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import nltk
# import re
#
#
# class Preprocess:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         # Read the dataset
#         self.df = pd.read_csv(self.file_path)
#         self.stop_words = set(stopwords.words('english'))
#
#         nltk.download('punkt')
#         nltk.download('wordnet')
#         nltk.download('stopwords')
#
#     def clean_data(self):
#         # Remove rows with missing 'text' or 'type' values
#         self.df = self.df.dropna(subset=['text', 'type'])
#
#         # Remove rows with empty or whitespace-only text
#         self.df = self.df[self.df['text'].str.strip().astype(bool)]
#
#         # Clean text (remove URLs, special characters, and convert to lowercase)
#         self.df['text'] = self.df['text'].apply(lambda x: unescape(x))
#         self.df['text'] = self.df['text'].apply(self._clean_text)
#
#         # Tokenize, lowercase, lemmatize, and remove stopwords
#         self.tokenize()
#         self.lowercase()
#         self.lemmatize()
#         self.remove_stopwords()
#
#     def split_data(self, test_size=0.3, random_state=20):
#         # Split the dataset into train and test sets after preprocessing
#         x_train, x_test, y_train, y_test = train_test_split(self.df['text'], self.df['type'],
#                                                             test_size=test_size, random_state=random_state)
#         return x_train, x_test, y_train, y_test
#
#     @staticmethod
#     def _clean_text(text):
#         # Remove non-ASCII characters
#         text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#
#         # Remove special characters except alphanumeric characters and spaces
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#
#         return text.lower()
#
#     def tokenize(self):
#         # Tokenize text
#         self.df['text'] = self.df['text'].apply(word_tokenize)
#
#     def lowercase(self):
#         # Convert tokens to lowercase
#         self.df['text'] = self.df['text'].apply(lambda x: [word.lower() for word in x])
#
#     def lemmatize(self):
#         # Lemmatize tokens
#         lemmatizer = WordNetLemmatizer()
#         self.df['text'] = self.df['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
#
#     def remove_stopwords(self):
#         # Remove stop words
#         self.df['text'] = self.df['text'].apply(lambda x: [word for word in x if word not in self.stop_words])
