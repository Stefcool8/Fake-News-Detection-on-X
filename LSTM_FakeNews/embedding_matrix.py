import numpy as np
import gensim


# For Word2Vec embeddings
class EmbeddingMatrix:
    def __init__(self, word2vec_model_path, tokenizer, max_words):
        self.word2vec_model_path = word2vec_model_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.embedding_dim = 300

    def create_embedding_matrix(self):
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=True)
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))

        missing_words = []

        for word, index in self.tokenizer.word_index.items():
            if word not in word_vectors.key_to_index:
                missing_words.append(word)
                continue
            if index >= self.max_words:
                continue
            if word in word_vectors:
                embedding_matrix[index] = word_vectors[word]

        return embedding_matrix, missing_words


# For GloVe embeddings
# class EmbeddingMatrix:
#     def __init__(self, glove_file_path, tokenizer, max_words):
#         self.glove_file_path = glove_file_path
#         self.tokenizer = tokenizer
#         self.max_words = max_words
#         self.embedding_dim = 100
#
#     def create_embedding_matrix(self):
#         # Load the GloVe model
#         embeddings_index = {}
#         with open(self.glove_file_path, encoding="utf-8") as f:
#             for line in f:
#                 values = line.split()
#                 word = values[0]
#                 coefs = np.asarray(values[1:], dtype='float32')
#                 embeddings_index[word] = coefs
#
#         embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
#
#         missing_words = []
#
#         for word, index in self.tokenizer.word_index.items():
#             if word not in embeddings_index:
#                 missing_words.append(word)
#                 continue
#             if index >= self.max_words:
#                 continue
#             embedding_matrix[index] = embeddings_index[word]
#
#         return embedding_matrix, missing_words
