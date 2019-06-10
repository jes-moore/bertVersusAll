# General
import pandas as pd
import numpy as np

# Data Prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from bert_serving.client import BertClient

import logging
log = logging.getLogger(__name__)

# Before running this module, you need to serve the BERT module
# bert-serving-start -model_dir ~/bert/models/cased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len=250


def read_and_cleanup_wiki_movie_plots(nrows=None):
    """
    Functiont hat processes the nlp dataset from kaggle:
    https://www.kaggle.com/
    jrobischon/wikipedia-movie-plots/downloads/wikipedia-movie-plots.zip/1

    Returns xtrain (the text for training) and classes (the multiclass labels)
            xtrain = movie_plot
            classes = origin
    """

    # def analyze_nlp_dataset():

    df = pd.read_csv(
        'data/wiki_movie_plots_deduped.csv',
        usecols=[0, 1, 2, 5, 7])
    df.columns = ['release_year', 'title', 'origin', 'genre', 'movie_plot']
    df = df[['movie_plot', 'origin']]
    df.columns = ['text', 'classes']
    log.info("Raw data loaded and prepared for processing")
    if nrows is not None:
        df = df.sample(n=nrows)
    return df


def preproc_text_bert(x, bc, maxlen):
    '''
    Function that accepts a text-string (x), a bert-client (bc) and
    a maxlen (maximum word-length for the bert-model) and produces
    the BERT encoding
    '''
    x = ' '.join(x.strip().split()[0:maxlen])
    return bc.encode([x])


def prepare_df_for_training(df, maxlen):
    '''
    Function to process a file that is formatted as follows:
      text  | classes
    string   class1,class2,class3
    returns a dataframe with bert-encoded sentences
    '''

    # Check Proper Columns
    if 'text' not in df.columns:
        raise ValueError("Missing Text Column")
    if 'classes' not in df.columns:
        raise ValueError("Missing Classes Column")

    # Get BERT Encoding
    bc = BertClient()  # Initialise Bert Client
    log.info("Encoding sentences using BERT, this will take several minutes")
    df['bert_enc'] = df['text'].apply(lambda x: preproc_text_bert(x, bc, maxlen))
    log.info("Encoding sentences complete")
    return df


def explode_classes(df):
    '''
    Function takes the output of prepare_df_for_keras and creates
    a label-encoded output of the classes column for training
    '''
    df.classes = df.classes.apply(lambda x: [cl.strip() for cl in x.split(',')])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df.classes)
    log.info("Classes expoded for multilabel classification")
    return df, y, mlb.classes_


def prep_training_data_keras(df, y):
    '''
    Function that takes a dataframe output from prepare_data_for_keras
    and the label-encoded output of explode classes to prepare training
    data for modelling
    '''
    # Extract Properly Formatted Matrix of Bert Enc
    X = np.stack(df.bert_enc.apply(lambda x: x[0]).values, axis=1).T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    log.info("Training and datasets ready for training")
    return X_train, X_test, y_train, y_test


def describe_training_data(y, classes):
    log.info(f"Training Examples = {y.shape[0]}")
    log.info(f"Num-classes = {y.shape[1]}")
    for ix, cl in enumerate(classes):
        log.info(f"Class = {cl}, Count = {np.sum([out[ix] for out in y])}")
    return
