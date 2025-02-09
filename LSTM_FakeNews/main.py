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

MAX_WORDS = 5000
BATCH_SIZE = 64
EPOCHS = 300


def hyperparameter_tuning(lstm_model, x_train, y_train, x_val, y_val):
    tuner = kt.Hyperband(
        lstm_model.hyperparameter_tuning,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='my_dir',
        project_name='intro_to_kt'
    )

    tuner.search(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]

    print(best_model)

    return best_model


def plot_word_cloud(tweets):
    # Join all tweets into a single string
    text = ' '.join([' '.join(tweet) for tweet in tweets])

    wordcloud = WordCloud(width=800, height=400).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # save word cloud to file
    wordcloud.to_file('wordcloud.png')


def main():
    # Read data from CSV
    df = pd.read_csv('tweets_dataset.csv')

    tweets = df['tweet'].tolist()
    targets = df['target'].tolist()
    majority_answers = df['3_label_majority_answer'].tolist()

    # Initialize label data class
    label_data = LabelData()

    # Take only 1000 samples for testing
    # tweets = tweets[:1000]
    # targets = targets[:1000]
    # majority_answers = majority_answers[:1000]

    # Get truthfulness and convert to binary
    truthfulness_values = [label_data.get_truthfulness(targ, maj_ans)
                           for targ, maj_ans in zip(targets, majority_answers)]
    binary_truthfulness_values = [label_data.convert_truthfulness_to_binary(val) for val in truthfulness_values]

    # clean tweets
    preprocess = Preprocess(sub_words_path='ninja_replacements.txt')
    cleaned_tweets, cleaned_labels = preprocess.filter_tweets(tweets, binary_truthfulness_values)

    # For testing only
    # cleaned_tweets, cleaned_labels = tweets, binary_truthfulness_values

    # Get maximum length of a tweet
    max_len = max([len(tweet) for tweet in cleaned_tweets])
    print(f"Maximum length of a tweet: {max_len}")
    print(f"Number of tweets: {len(cleaned_tweets)}")

    # Tokenize and pad sequences for LSTM training
    tokenizer = Tokenizer(num_words=MAX_WORDS)

    # Fit the tokenizer on the cleaned tweets to establish the initial vocabulary
    tokenizer.fit_on_texts(cleaned_tweets)

    print(f"Vocabulary size: {len(tokenizer.word_index)}")

    # Save replacements to file
    # save_replacements_to_file('GoogleNews-vectors-negative300.bin', cleaned_tweets, 'ninja_replacements.txt')
    # save_replacements_to_file('glove.6B.100d.txt', 'glove', cleaned_tweets, 'ninja_replacements_glove.txt')

    # Tokenize and pad sequences with replaced tweets
    sequences = tokenizer.texts_to_sequences(cleaned_tweets)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # # # Check for missing words in Word2Vec model
    # # missing_words = get_sub_words('GoogleNews-vectors-negative300.bin', tokenizer)
    # # # missing_words = check_missing_words('glove.twitter.27B.200d.txt', tokenizer)
    # # print(f"Number of missing words: {len(missing_words)}")
    # # print(f"Missing words: {missing_words}")
    #
    # Convert data to numpy arrays to ensure compatibility with TensorFlow
    x = np.array(padded_sequences)
    y = np.array(cleaned_labels)

    # Split data into training and validation sets with stratification
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Verify shapes before training
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

    print(f"Number of positive samples in test set: {np.sum(y_val)}")
    print(f"Number of negative samples in test set: {len(y_val) - np.sum(y_val)}")

    # Load word2vec and create embedding matrix
    embedding_matrix_creator = EmbeddingMatrix('GoogleNews-vectors-negative300.bin', tokenizer, max_words=MAX_WORDS)
    # embedding_matrix_creator = EmbeddingMatrix('glove.6B.100d.txt', tokenizer, MAX_WORDS)
    embedding_matrix, missing_words = embedding_matrix_creator.create_embedding_matrix()

    print(missing_words)

    # Save extra replacements to file
    # save_replacements_to_file('GoogleNews-vectors-negative300.bin', missing_words, 'ninja_replacements_second.txt')
    # save_replacements_to_file('glove.6B.100d.txt', 'glove', missing_words, 'ninja_replacements_glove_second.txt')

    # Initialize and build LSTM model
    lstm_model = LSTMModel(embedding_matrix, max_words=MAX_WORDS, max_len=max_len)
    lstm_model.build_model()

    # Initialize and build CNN model
    # cnn_model = CNNModel(embedding_matrix, max_words=MAX_WORDS, max_len=max_len)
    # cnn_model.build_model()

    # Initialize and build Hybrid model
    # hybrid_model = HybridModel(embedding_matrix, max_words=MAX_WORDS, max_len=max_len)
    # hybrid_model.build_model()

    # Load model
    # lstm_model.load_model('saved_models/lstm_model.keras')

    # Train LSTM model
    history = lstm_model.train_model(x_train, y_train, x_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Train CNN model
    # history = cnn_model.train_model(x_train, y_train, x_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Train Hybrid model
    # history = hybrid_model.train_model(x_train, y_train, x_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Save LSTM model
    lstm_model.save_model('saved_models/lstm_model.keras')

    # Save CNN model
    # cnn_model.save_model('saved_models/cnn_model.keras')

    # Save Hybrid model
    # hybrid_model.save_model('saved_models/hybrid_model.keras')

    # Evaluate the LSTM model
    y_pred, y_pred_prob = lstm_model.evaluate(x_val)

    # Evaluate the CNN model
    # y_pred, y_pred_prob = cnn_model.evaluate(x_val)

    # Evaluate the Hybrid model
    # y_pred, y_pred_prob = hybrid_model.evaluate(x_val)

    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    # Calculate and print the percentage of 0 and 1 classes predicted
    unique, counts = np.unique(y_pred, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    total_predictions = sum(class_distribution.values())
    for cls, count in class_distribution.items():
        print(f"Class {cls}: {count} predictions ({(count / total_predictions) * 100:.2f}%)")

    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print LSTM model summary
    lstm_model.model.summary()

    # Print CNN model summary
    # cnn_model.model.summary()

    # Print Hybrid model summary
    # hybrid_model.model.summary()

    # Find misclassified tweets
    misclassified_tweets = []
    misclassified_labels = []

    for i in range(len(y_val)):
        if y_val[i] != y_pred[i]:
            misclassified_tweets.append(cleaned_tweets[i])
            misclassified_labels.append(y_val[i])

    # Save misclassified tweets to CSV
    with open('misclassified_tweets.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet', 'label'])
        for tweet, label in zip(misclassified_tweets, misclassified_labels):
            writer.writerow([tweet, label])


if __name__ == "__main__":
    main()
