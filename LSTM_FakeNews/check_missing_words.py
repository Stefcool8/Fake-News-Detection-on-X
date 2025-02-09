import concurrent.futures

import numpy as np
from gensim.models import KeyedVectors
from spellchecker import SpellChecker
import wordninja


def process_word(word, word_vectors, spell):
    if word in word_vectors:
        print("Word already in vocabulary: ", word)
        return None

    original_word = word
    corrected = spell.correction(word)

    print("----")
    print("Initial word: ", word)

    if corrected and corrected in word_vectors:
        print("Corrected word: ", corrected)
        return original_word, corrected

    if corrected:
        print("Corrected word: ", corrected)
        word = corrected

    splitted = wordninja.split(word)
    repaired = []

    # if splitted contains the original_word, then correction and splitting are creating a loop
    if original_word in splitted:
        print("Valid word not found in vocabulary: ", word)
        return original_word, "0"
    else:
        for word in splitted:
            if len(word) < 2:
                continue
            if word not in word_vectors:
                result = process_word(word, word_vectors, spell)
                if result:
                    _, repaired_word = result
                    repaired.extend(repaired_word.split())
                else:
                    repaired.append(word)
            else:
                repaired.append(word)

    if repaired:
        repaired_words = " ".join(repaired)
        print("Repaired: ", repaired)
        print("----")
        return original_word, repaired_words

    return None


def load_word_vectors(word_vector_path, word_vector_type):
    if word_vector_type == "word2vec":
        return KeyedVectors.load_word2vec_format(word_vector_path, binary=True)
    elif word_vector_type == "glove":
        embeddings_index = {}
        with open(word_vector_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    else:
        raise ValueError("Invalid word vector type")


def get_sub_words(word_vector_path, word_vector_type, tokenized_tweets):
    word_vectors = load_word_vectors(word_vector_path, word_vector_type)
    spell = SpellChecker()
    replacements = {}

    # words_to_process = set(word for tweet in tokenized_tweets for word in tweet)
    words_to_process = set(word for word in tokenized_tweets)
    print(f"Number of words to process: {len(words_to_process)}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_word = {executor.submit(process_word, word, word_vectors, spell): word for word in words_to_process}
        for future in concurrent.futures.as_completed(future_to_word):
            result = future.result()
            if result:
                original_word, replacement = result
                replacements[original_word] = replacement

    return replacements


def save_replacements_to_file(word_vector_path, word_vector_type, tokenized_tweets, file_path):
    replacements = get_sub_words(word_vector_path, word_vector_type, tokenized_tweets)

    with open(file_path, 'w') as f:
        for key, value in replacements.items():
            f.write(f"{key} {value}\n")


# Below code is for fixing 'covid' replacements, by replacing it with 'coronavirus'
# Otherwise, 'covid' will be replaced mostly with 'cov id'

# if __name__ == "__main__":
#     word_vectors = load_word_vectors("glove.6B.100d.txt", "glove")
#     spell = SpellChecker()
#
#     sub_words_dict = {}
#     with open('ninja_replacements_glove.txt', 'r') as f:
#         for line in f:
#             # First word is the key, the remaining words are the value
#             words = line.split()
#             value = " ".join(words[1:])
#
#             # if words[0] contains 'covid', replace 'covid' with 'coronavirus'
#             if 'covid' in words[0]:
#                 key = words[0]
#                 key = key.replace('covid', 'coronavirus')
#
#                 result = process_word(key, word_vectors, spell)
#                 if result:
#                     _, replacement = result
#                     value = replacement
#
#             sub_words_dict[words[0]] = value
#
#     with open('new_ninja_replacements_glove.txt', 'w') as f:
#         for key, value in sub_words_dict.items():
#             f.write(f"{key} {value}\n")


# def check_missing_words(glove_file_path, tokenizer):
#     missing_words = set()
#
#     # Load GloVe embeddings
#     embeddings_index = {}
#     with open(glove_file_path, encoding="utf-8") as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             embeddings_index[word] = True
#
#     # Check for missing words
#     for word, index in tokenizer.word_index.items():
#         if word not in embeddings_index:
#             missing_words.add(word)
#
#     return missing_words
