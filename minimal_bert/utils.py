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


def load_bert_and_prep_for_training(n_samples):
    with open('data/bert_df.pickle', 'rb') as f:
        df = pickle.load(f)
    if n_samples is not None:
        df = df.sample(n=n_samples)
    df, y, classes = pp.explode_classes(df)
    pp.describe_training_data(y, classes)
    X_train, X_test, y_train, y_test = pp.prep_training_data_keras(df, y)
    return X_train, X_test, y_train, y_test
