# import pandas as pd
#
# df = pd.read_csv('tweets_dataset.csv')
# print(df.head())

# import re, collections
#
#
# def get_stats(vocab):
#     pairs = collections.defaultdict(int)
#     for word, freq in vocab.items():
#         symbols = word.split()
#         for i in range(len(symbols) - 1):
#             pairs[symbols[i], symbols[i + 1]] += freq
#     return pairs
#
#
# def merge_vocab(pair, v_in):
#     v_out = {}
#     bigram = re.escape(' '.join(pair))
#     p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#     for word in v_in:
#         w_out = p.sub(''.join(pair), word)
#         v_out[w_out] = v_in[word]
#     return v_out
#
#
# def byte_pair_encoding(vocab):
#     pairs = get_stats(vocab)
#
#     while len(pairs.values()) != 0:
#         # Max value
#         Max = max(list(pairs.values()))
#
#         # Find the key(s) that correspond to Max
#         best_pair = []
#         for key, value in pairs.items():
#             if value == Max:
#                 best_pair.append(key)
#
#         # Pair the most frequent pairs
#         for pair in best_pair:
#             vocab = merge_vocab(pair, vocab)
#             # print(pair,':',Max)
#         pairs = get_stats(vocab)
#     return vocab.keys()
#
#
# byte_pair_encoding(res_dict)

# import wordninja
# from gensim.models import KeyedVectors
#
# word2vec_model_path = "GoogleNews-vectors-negative300.bin"
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
# splitted = wordninja.split("coronavirus")
# for word in splitted:
#     print(word)
#
# if 'coronavirus' not in word_vectors.key_to_index:
#     print("Word not in Word2Vec model")
