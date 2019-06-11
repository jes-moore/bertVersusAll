import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import logging
log = logging.getLogger(__name__)


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
    df = df.sample(frac=1)
    df.columns = ['release_year', 'title', 'origin', 'genre', 'movie_plot']
    df = df[['movie_plot', 'origin']]
    df.columns = ['text', 'y']
    log.info("Raw data loaded and prepared for processing")
    if nrows is not None:
        df = df.sample(n=nrows)
    return df


def preproc_text(x):
    '''
    Text preprocessing pipeline that returns
    a processed string
    '''
    regex = re.compile(r'[►*/-@.?!&~,":#$;\'()=+|0-9]')  # Remove Characters
    x = regex.sub("", x)  # Run Regex
    x = x.replace('  ', ' ')  # Replace Double Spaces
    x = x.replace('-', ' ')  # Replace Hyphens

    return x


def load_and_prep_training_data(max_features, maxlen):

    # n_columns = max_features (25000, 50k, 75k)
    # n_rows = dimensions of embedding (300)
    # maxlen = # of words input text

    '''
    Processes two files, train.csv and test.csv.
        1. Loads Data from JSON
        2. Preproces Text
        3. Split into X (text) and Y
        4. Tokenizes the sentences and converts them to sequences
        5. Pads the sequences/truncates (front/pre padding) to the fixed maxlength
        6. Returns X, Y, and the word-list
    '''

    # 1. Load Text
    df = read_and_cleanup_wiki_movie_plots()
    log.info(f"Data shape : {df.shape}")

    # 2. Preprocess Text
    log.info("Preprocessing sentences...")
    df.text = df.text.apply(preproc_text)

    # 3. Split X and Y into Arrays
    log.info("Splitting into X and Y for training...")
    df_X = df["text"].values
    df_Y = df['y'].values
    mlb = MultiLabelBinarizer()
    df_Y = mlb.fit_transform(df_Y)
    log.info("Classes expoded for multilabel classification")

    # 4. Tokenize the sentences
    log.info('Tokenizing the sentences...')
    tokenizer = Tokenizer()  # ['this', 'is', 'a', 'cat'] 'string'.split(' ')
    tokenizer.fit_on_texts(list(df_X))
    df_X = tokenizer.texts_to_sequences(df_X)  # [0, 0, 0, 0, 0, 2542, 2504, 12, 123]

    # Remove Feature-indices above max_features
    df_X = [[x for x in li if x < max_features] for li in df_X]

    # 5. Pad the sentences
    log.info('Padding sequences (pre) for input data...')
    df_X = pad_sequences(df_X, maxlen=maxlen, padding='pre')

    log.info('Complete - Returning Values')
    return df_X, df_Y, tokenizer.word_index  # vocubulary
