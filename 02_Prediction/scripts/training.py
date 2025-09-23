import os
import datetime
ISOTIMEFORMAT = '%m-%d %H:%M'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np

import tensorflow as tf

np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from scipy.stats import spearmanr, pearsonr

from sklearn.model_selection import KFold



def one_hot_encode(df, col='utr', reverse = True,  seq_len=50):
    #TBD adding truncate direction forward or reverse.
    # Dictionary returning one-hot encoding of nucleotides.
    # reverse = True is for 5UTR, splice the seq from 3'; reverse = False is 3UTR.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col]):
        if isinstance(seq, float) and np.isnan(seq):
            seq = ('n' * seq_len)
        seq = seq.lower()
        if len(seq) < seq_len:
            if reverse:
                seq = ('n' * (seq_len - len(seq))) + seq
            else:
                seq = seq + ('n' * (seq_len - len(seq)))

        if len(seq) > seq_len:
            if reverse:
                seq = seq[-seq_len:]
            else:
                seq = seq[:seq_len]
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def RNN_model(x, y, border_mode='same', inp_len=50, nodes=40, layers=3, filter_len=8, nbr_filters=120,
                dropout1=0, dropout2=0, dropout3=0, nb_epoch=3):
    ''' Build model archicture and fit.'''

    model = Sequential()
    if layers >= 1:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
    if layers >= 2:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model.add(Dropout(dropout1))
    if layers >= 3:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model.add(Dropout(dropout2))
    model.add(LSTM(units=nbr_filters))
    model.add(Dropout(dropout3))
    #model.add(Flatten())

    model.add(Dense(nodes))
    model.add(Activation('relu'))
    model.add(Dropout(dropout3))
    #
    # model.add(Dense(nodes))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout3))

    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mean_absolute_error', optimizer=adam)
    #model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit(x, y, batch_size=128, epochs=nb_epoch, verbose=1)
    print("nodes: ", nodes)
    print("Filter:", nbr_filters)
    return model