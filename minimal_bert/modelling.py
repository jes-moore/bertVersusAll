import keras
import keras_metrics
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from minimal_bert.preproc import preproc_text_bert

import logging
log = logging.getLogger(__name__)


def train_and_return_model(X_train, X_test, y_train, y_test, n_hidden):
    '''
    Trains a generic model based on the BERT encodings and a specified
    number of units in the final layer to train a multiclass prediction model
    '''
    # Model Design
    inp = Input(shape=(768,))
    x = Dense(n_hidden, activation="relu")(inp)
    x = Dense(y_train.shape[1], activation="sigmoid")(x)

    # Optimizer and Early Stopping
    adam = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    # Compile Model
    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()],
    )

    # Run Model
    history = model.fit(
        X_train,
        y_train,
        batch_size=200,
        epochs=10000,
        callbacks=[es],
        validation_data=(X_test, y_test),
        verbose=0
    )
    return model, history


def predict_text(text, bc, maxlen, model):
    enc = preproc_text_bert(text, bc, maxlen)
    score = model.predict(enc)
    return score
