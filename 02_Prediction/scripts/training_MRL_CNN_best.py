import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import tensorflow as tf

np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from scipy.stats import spearmanr

def train_model(x, y, border_mode='same', inp_len=50, nodes=40, layers=3, filter_len=8, nbr_filters=120,
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
    model.add(Flatten())

    model.add(Dense(nodes))
    model.add(Activation('relu'))
    model.add(Dropout(dropout3))

    model.add(Dense(nodes/2))
    model.add(Activation('relu'))
    model.add(Dropout(dropout3))

    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mean_absolute_error', optimizer=adam)

    model.fit(x, y, batch_size=128, epochs=nb_epoch, verbose=1)
    print("nodes: ", nodes)
    print("Filter:", nbr_filters)
    return model


def test_data(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))

    # Make predictions
    predictions = model.predict(test_seq).reshape(-1, 1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    #df.loc[:, output_col] = predictions
    return df

def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2

data_file = '../data/Hek_R100bp5UTR_RNA5Ribo1.csv'
df = pd.read_csv(data_file)
#df = pd.read_csv('../data/GSM3130435_egfp_unmod_1.csv')

#df.sort_values('total_reads', inplace=True, ascending=False)
df = df.sample(frac=1)

seq_length = 100

df.reset_index(inplace=True, drop=True)
#df['rl'] = df['rl'].apply(np.log2)
#df = df.loc[:280000]

# The training set has 260k UTRs and the test set has 20k UTRs.
e_test = df.loc[:1500]
e_train = df.loc[1500:]

# One-hot encode both training and test UTRs
seq_e_train = one_hot_encode(e_train, seq_len=seq_length)
seq_e_test = one_hot_encode(e_test, seq_len=seq_length)

# Scale the training mean ribosome load values
e_train.loc[:,'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,'rl'].values.reshape(-1,1))

#e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:, 'rl'].reshape(-1, 1))

model = train_model(seq_e_train, e_train['scaled_rl'], nb_epoch=30, border_mode='same',
                   inp_len=seq_length, nodes=80, layers=3, nbr_filters=240, filter_len=8, dropout1=0,
                   dropout2=0, dropout3=0.2)
model.save('./saved_models/my_special_model.hdf5')
e_pred = test_data(df=e_test, model=model, test_seq=seq_e_test, obs_col='rl')
r = r2(e_pred['rl'], e_pred['pred'])
sp_cor = spearmanr(e_pred['rl'], e_pred['pred'])
print('Data File: ', data_file)
print('r-squared = ', r)
print('spearman R = ', sp_cor[0])
e_pred.to_csv("./test_prediction_R.csv")

e_pred = test_data(df=e_train, model=model, test_seq=seq_e_train, obs_col='rl')
r = r2(e_train['rl'], e_train['pred'])
sp_cor = spearmanr(e_pred['rl'], e_pred['pred'])
print('Training set r-squared = ', r)
print('spearman R = ', sp_cor[0])
e_pred.to_csv("./train_prediction_R.csv")

