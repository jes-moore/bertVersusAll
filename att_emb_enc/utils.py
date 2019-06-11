from att_emb_enc import preproc
from att_emb_enc import embed
import pickle
import os

import logging
log = logging.getLogger(__name__)


def create_or_load_preproc(max_features, maxlen, reload=False):
    """
    Checks if the preprocessing settings have already been ran and the output
    files already created.

    If not, then it runs the preprocessing steps to create the output
    file.
    """
    pickle_f = "data/processed_inputs/X_Y_word_index_maxfeat{}_maxlen{}.pkl"\
               .format(max_features, maxlen)
    if reload | (not os.path.exists(pickle_f)):
        X, Y, vocabulary = preproc.load_and_prep_training_data(max_features, maxlen)
        with open(pickle_f, "wb") as f:
            pickle.dump((X, Y, vocabulary), f)
        log.info("Input Data Saved to {}".format(pickle_f))
    else:
        with open(pickle_f, "rb") as f:
            X, Y, vocabulary = pickle.load(f)
        log.info("Input Data Loaded From {}".format(pickle_f))
    return X, Y, vocabulary


def create_or_load_embeddings(vocabulary, max_features, embed_size, reload=False):
    """
    Sister function to create_or_load_preproc.
    Checks if the embedding file has been crated for the specified settings

    If not, then it runs the creation of averaged word embeddings based on
    input settings
    """
    pickle_f = "data/processed_embeddings/avg_embeddings_maxfeat{}.pkl"\
               .format(max_features)

    if reload | (not os.path.exists(pickle_f)):
        embed_matrix = embed.load_all_embeddings(vocabulary, max_features, embed_size)
        with open(pickle_f, "wb") as f:
            pickle.dump(embed_matrix, f)
            log.info("Embeddings Saved to {}\n".format(pickle_f))
    else:
        with open(pickle_f, "rb") as f:
            embed_matrix = pickle.load(f)
        log.info("Embeddings File Loaded From {}\n".format(pickle_f))
    return embed_matrix


def extract_validation_results(history):
    loss = history.history['val_loss'][-1]
    acc = history.history['val_acc'][-1]
    prec = history.history['val_precision'][-1]
    rec = history.history['val_recall'][-1]
    return loss, acc, prec, rec


def calc_f1(prec, rec):
    if prec + rec == 0:
        return None
    return (2 * prec * rec) / (prec + rec)
