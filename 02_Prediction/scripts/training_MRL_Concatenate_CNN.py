import os
import datetime
ISOTIMEFORMAT = '%m-%d %H:%M'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np
import math
import scipy.stats as stats
from sklearn import preprocessing
import tensorflow as tf
import training as train
np.random.seed(1337)

from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import concatenate
from keras.layers.recurrent import LSTM
from scipy.stats import spearmanr, pearsonr

from sklearn.model_selection import KFold

def train_model(x, y, border_mode='same', inp_len=50, nodes=40, layers=3, filter_len=8, nbr_filters=120,
                dropout1=0, dropout2=0, dropout3=0, nb_epoch=3):
    ''' Build model archicture and fit.'''

    model1 = Sequential()
    if layers >= 1:
        model1.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
    if layers >= 2:
        model1.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model1.add(Dropout(dropout1))
    if layers >= 3:
        model1.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model1.add(Dropout(dropout2))
    model1.add(Flatten())
    #model1.add(LSTM(units=32, return_sequences=False))

    model2 = Sequential()
    if layers >= 1:
        model2.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
    if layers >= 2:
        model2.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model2.add(Dropout(dropout1))
    if layers >= 3:
        model2.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model2.add(Dropout(dropout2))
    model2.add(Flatten())
    #model2.add(LSTM(units=32, return_sequences=False))

    # merged_model = Sequential()
    # #merged_model = concatenate([model1.output, model2.output])
    # merged_model.add(concatenate([model1, model2], axis=1))
    # merged_model.add(Dense(nodes))
    # merged_model.add(Activation('relu'))
    # merged_model.add(Dropout(dropout3))
    # merged_model.add(Dense(1))
    # merged_model.add(Activation('linear'))
    concatenated = concatenate([model1.output, model2.output])
    out = Dense(nodes, activation='relu')(concatenated)
    out = Dropout(dropout3)(out)
    out = Dense(1, activation= 'linear')(out)
    merged_model = Model(inputs=[model1.inputs, model2.inputs], outputs=[out])
    # compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    merged_model.compile(loss='mean_absolute_error', optimizer=adam)
    #model.compile(loss='mean_squared_error', optimizer=adam)
    history = merged_model.fit(x, y, batch_size=128, epochs=nb_epoch, verbose=1)
    print("nodes: ", nodes)
    print("Filter:", nbr_filters)
    return [merged_model, history]


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
    for i, seq in enumerate(df[col]):
        if isinstance(seq, float) and np.isnan(seq):
            seq = ('n' * seq_len)
        seq = seq.lower()
        if len(seq) < seq_len:
            seq = ('n' * (seq_len - len(seq))) + seq
        if len(seq) > seq_len:
            seq = seq[-seq_len:]
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2

seq_length = 100
np_epoch = 40
nodes = 40
nbr_filters = 120
filter_len = 8
fold = 5
r2_results = np.zeros(fold)
pearsonr_results = np.zeros(fold)
sp_cor_results = np.zeros(fold)
cell_line = 'HEK293'  #hek muscle pc3
#data_file = '../data/All_R200L200UTR_CDS_M5R0.1.csv'
#data_file = '../data/5UTR+CDS+3UTR 40260.csv'
data_file = '../data/mRNA_HL_Hela+HEK_5UTR200_3UTR300.csv'
best_prediction = 0
train_history = []
pred_df = pd.DataFrame()
df = pd.read_csv(data_file)
#df = df.drop(columns = ['cds'])
#df = pd.read_csv('../data/GSM3130435_egfp_unmod_1.csv')

if cell_line !='all':
    df = df.loc[df['CellLine'] == cell_line]
df = df.sample(frac=1)
sample_size = len(df)
df.reset_index(inplace=True, drop=True)
# log2 can improve the r2 performance but not spearman cor.
df['rl'] = df['rl'].apply(np.log2)
KF = KFold(n_splits=fold, shuffle=False)
fold_no  = 0
for train_index,test_index in KF.split(df):

    print('Fold: ', fold_no+1)
    e_train = df.iloc[train_index, :]
    e_test = df.iloc[test_index, :]
    # One-hot encode both training and test UTRs
    #seq_e_train = one_hot_encode(e_train, seq_len=seq_length)
    seq1_train = train.one_hot_encode(e_train, col='5utr', seq_len=seq_length)
    seq2_train = train.one_hot_encode(e_train, col='3utr', reverse=False, seq_len=seq_length)
    model_input = [seq1_train, seq2_train]
    #seq_e_test = one_hot_encode(e_test, seq_len=seq_length)
    seq1_test = train.one_hot_encode(e_test, col='5utr', seq_len=seq_length)
    seq2_test = train.one_hot_encode(e_test, col='3utr', reverse=False, seq_len=seq_length)
    model_test = [seq1_test, seq2_test]



    # Scale the training mean ribosome load values
    e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,'rl'].values.reshape(-1,1))
    #e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:, 'rl'].reshape(-1, 1))

    model, history = train_model(model_input, e_train['scaled_rl'], nb_epoch=np_epoch, border_mode='same',
                       inp_len=seq_length, nodes=nodes, layers=3, nbr_filters=nbr_filters, filter_len=filter_len,
                        dropout1=0, dropout2=0, dropout3=0.2)

    e_pred = test_data(df=e_test, model=model, test_seq=model_test, obs_col='rl')
    r = r2(e_pred['rl'], e_pred['pred'])
    pearson_r = pearsonr(e_pred['rl'], e_pred['pred'])
    sp_cor = spearmanr(e_pred['rl'], e_pred['pred'])

    if sp_cor[0] > best_prediction:
        best_prediction = sp_cor[0]
        model.save('./saved_models/Temp_best_Model.hdf5')
        train_history.append(history)
    r2_results[fold_no] = r
    pearsonr_results[fold_no] = pearson_r[0]
    sp_cor_results[fold_no] = sp_cor[0]
    pred_df = pred_df.append(e_pred)
    fold_no = fold_no + 1
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
model = tf.keras.models.load_model('./saved_models/Temp_best_Model.hdf5')
model.save('./saved_models/'+ cell_line + '-len'+str(seq_length)+ '-Sp'+ str(round(sp_cor_results.max(), 3)) + '-merge-CNN' + theTime + '.hdf5')
#pred_df.to_csv('../data/Muscle_prediction.csv')
for score in r2_results:
    print('r-squared = ', score)
for score in pearsonr_results:
    print('pearson r = ', score)
for score in sp_cor_results:
    print('spearman R = ', score)
print('Cell Line:', cell_line, ' Sample Size:', sample_size)
print('Data File:', data_file, ' Seq Length:' + str(seq_length))
print('epoch =', str(np_epoch), ' nodes:', str(nodes), ' nbr_filters:', nbr_filters, ' filter_len:',filter_len)
print('r-squared Mean= ', str(r2_results.mean()))
print('pearsonr R Mean= ', str(pearsonr_results.mean()))
print('spearman R Mean= ', str(sp_cor_results.mean()))

