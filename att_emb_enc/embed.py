import os
import gc
import numpy as np

import logging
log = logging.getLogger(__name__)


def load_embedding(embed_file, vocabulary, max_features, embed_size):
    '''
    Function that creates a dictionary-lookup table
    from a specified embedding file.
    '''

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if 'glove' in embed_file:
        embed_file = 'data/embeddings/glove.840B.300d.txt'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o
                                in open(embed_file))
    if 'wiki' in embed_file:
        embed_file = 'data/embeddings/wiki-news-300d-1M.vec'
        embeddings_index = dict(get_coefs(*o.rstrip().split(" ")) for o
                                in open(embed_file) if len(o) > 100)
    if 'paragram' in embed_file:
        embed_file = 'data/embeddings/paragram_300_sl999.txt'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o
                                in open(embed_file, encoding="utf8", errors='ignore')
                                if len(o) > 100)  # Skips First Line with embed size
    if 'Google' in embed_file:
        from gensim.models import KeyedVectors
        embed_file = 'data/embeddings/GoogleNews-vectors-negative300.bin'
        embeddings_index = KeyedVectors.load_word2vec_format(embed_file, binary=True)

        # Random Initialise of Embedding Matrix with small numbers
        # all_embs.mean(), all_embs.std() doesn't work on KeyedVectors
        num_words = min(max_features, len(vocabulary))
        embedding_matrix = (np.random.rand(num_words, embed_size) - 0.5)/5

        # Iterate
        for word, i in vocabulary.items():
            if i >= max_features:
                # Skip if n > specified # of max-features
                continue
            if word in embeddings_index:
                embedding_vector = embeddings_index.get_vector(word)
                embedding_matrix[i] = embedding_vector
        # Cleanup and Free Memory
        del embeddings_index
        gc.collect()

        return embedding_matrix

    # Read Embeddings for Specified File
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(max_features, len(vocabulary))

    # Not all words are contained in every pre-train
    # So we initialize the matrix with normally distributed variables from data
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in vocabulary.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    del embeddings_index
    gc.collect()
    return embedding_matrix


def load_all_embeddings(vocabulary, max_features, embed_size):
    '''
    Function that creates an averaged embedding matrix across
    a set of fixed embeddings.
        vocabulary = input vocabulary
        max_features = max number of words to retain
        embed_size = fixed embedding size for all files
        returns: averaged embedding matrix
    '''
    # Loop over embeddings and create output file
    embed_list = os.listdir('data/embeddings/')
    embedding_matrices = []
    for embed in embed_list:
        embedding_matrices.append(
            load_embedding(embed, vocabulary, max_features, embed_size))
        log.info('Loaded %s' % embed)
    # Average embedding matrice
    average_embedding_matrix = np.mean(embedding_matrices, axis=0)
    return average_embedding_matrix
