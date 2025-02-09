# Evaluate LSTM model

import pandas as pd
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.api.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocess import Preprocess
from label_data import LabelData
from embedding_matrix import EmbeddingMatrix
from lstm_model import LSTMModel
from cnn_model import CNNModel
from hybrid_model import HybridModel
from check_missing_words import get_sub_words, save_replacements_to_file

import keras_tuner as kt
import csv

# Load model
lstm_model = LSTMModel(None)
lstm_model.load_model('past_models/LSTM/best_accuracy_9.464')

preprocess = Preprocess(sub_words_path='ninja_replacements.txt')

tweet = "I am a very honest person."
cleaned_tweets, cleaned_labels = preprocess.filter_tweets([tweet], [1])

# Tokenize and pad sequences for LSTM training
tokenizer = Tokenizer(num_words=5000)

# Fit the tokenizer on the cleaned tweets to establish the initial vocabulary
tokenizer.fit_on_texts(cleaned_tweets)

# Tokenize and pad sequences with replaced tweets
sequences = tokenizer.texts_to_sequences(cleaned_tweets)
padded_sequences = pad_sequences(sequences, maxlen=46, padding='post')
#
# Convert data to numpy arrays to ensure compatibility with TensorFlow
x = np.array(padded_sequences)
y = np.array(cleaned_labels)

# Evaluate the model
y_pred, y_pred_prob = lstm_model.evaluate(x)

# Print the classification report
print(classification_report(y, y_pred))

# Print the confusion matrix
print(confusion_matrix(y, y_pred))
