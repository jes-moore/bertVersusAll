from minimal_bert import preproc as pp
import pickle

import logging
log = logging.getLogger(__name__)


def prepare_and_save_bert_encoded_df(maxlen):
    # Prepare Data For Training
    df = pp.read_and_cleanup_wiki_movie_plots()
    df = pp.prepare_df_for_training(df, maxlen=250)
    # Save to Pickle
    with open('data/bert_df.pickle', 'wb') as f:
        pickle.dump(df, f)
    log.info("Bert encoded df save to pickle")
    return


def load_bert_and_prep_for_training(n_samples, test_size):
    with open('data/bert_df.pickle', 'rb') as f:
        df = pickle.load(f)
    if n_samples is not None:
        df = df.sample(n=n_samples)
    df, y, classes = pp.explode_classes(df)
    pp.describe_training_data(y, classes)
    X_train, X_test, y_train, y_test = pp.prep_training_data_keras(df, y, test_size)
    return X_train, X_test, y_train, y_test


def extract_validation_results(history):
    loss = history.history['val_loss'][-1]
    acc = history.history['val_acc'][-1]
    prec = history.history['val_precision'][-1]
    rec = history.history['val_recall'][-1]
    return loss, acc, prec, rec
