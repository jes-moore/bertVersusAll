# General
import pandas as pd
import numpy as np

# Data Prep
from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Keras
import keras
import keras_metrics
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping


def prep_training_data_keras(df, y):
    '''
    Function that takes a dataframe output from prepare_data_for_keras
    and the label-encoded output of explode classes to prepare training
    data for modelling
    '''
    # Extract Properly Formatted Matrix of Bert Enc
    X = np.stack(df.bert_enc.apply(lambda x: x[0]).values, axis=1).T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test
