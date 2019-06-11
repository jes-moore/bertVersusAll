from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Bidirectional
from keras.models import Model
import keras_metrics
import keras
from keras.callbacks import EarlyStopping
from att_emb_enc.custom_layers import Attention

import logging
log = logging.getLogger(__name__)


def train_and_return_model(
        X_train, X_test, y_train,
        y_test, embedding_matrix, maxlen,
        max_features, embed_size, n_hidden,
        activation):
    '''
    Trains a generic model based on the BERT encodings and a specified
    number of units in the final layer to train a multiclass prediction model
    '''

    # Model Architecture
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(n_hidden, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(y_train.shape[1], activation=activation)(x)

    # Optimizer and Early Stopping
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=3)

    # Compile Model
    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()],
    )

    # Run Model
    history = model.fit(
        X_train,
        y_train,
        batch_size=100,
        epochs=10000,
        callbacks=[es],
        validation_data=(X_test, y_test),
        verbose=0
    )
    return model, history
