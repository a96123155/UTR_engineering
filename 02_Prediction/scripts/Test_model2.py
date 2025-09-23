import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
import random

import keras
np.random.seed(1337)
from keras.models import load_model
# from keras.preprocessing import sequence
# from keras.optimizers import RMSprop
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.layers.core import Dropout
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.convolutional import Conv1D

def test_data(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].reshape(-1, 1))

    # Make predictions
    predictions = model.predict(test_seq).reshape(-1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df
def one_hot_encode(df, col='utr', seq_len=100):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        if len(seq)< 50 :
            seq = 'n'*(50-len(seq)) + seq
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


# save scaler for different models
scaler_dic = {}

df = pd.read_csv('../data/Hek_R200bp5UTR_RNA5Ribo0.1.csv')
df['rl'] = df['rl'].apply(np.log2)
scaler = preprocessing.StandardScaler()
scaler.fit(df['rl'].values.reshape(-1,1))
scaler_dic['hek'] = scaler

df = pd.read_csv('../data/PC3_R200bp5UTR_RNA5Ribo0.1.csv')
df['rl'] = df['rl'].apply(np.log2)
scaler = preprocessing.StandardScaler()
scaler.fit(df['rl'].values.reshape(-1,1))
scaler_dic['pc3'] = scaler

df = pd.read_csv('../data/Muscle_R200bp5UTR_RNA5Ribo0.1.csv')
df['rl'] = df['rl'].apply(np.log2)
scaler = preprocessing.StandardScaler()
scaler.fit(df['rl'].values.reshape(-1,1))
scaler_dic['muscle'] = scaler

df = pd.read_csv('../data/GSM3130435_egfp_unmod_1.csv')
df.sort_values('total_reads', ascending=False).reset_index(drop=True)
#scale_utrs = df[:40000]
scaler = preprocessing.StandardScaler()
scaler.fit(df['rl'].values.reshape(-1,1))
scaler_dic['egfp'] = scaler



df = pd.read_csv('../data/RSV UTR Seeds.csv')

seq_length = 50

e_test = df

#seq_e_train = one_hot_encode(e_train, seq_len=seq_length)
seq_e_test = one_hot_encode(e_test, col='UTR 50bp mutated', seq_len=seq_length)

model_dic = {
    'hek_1': load_model('../modeling/saved_models/hek-len50-Sp0.745-04-07 22:50RNN.hdf5'),
    'hek_2': load_model('../modeling/saved_models/hek-len50-Sp0.732-04-07 03:11RNN.hdf5'),
    'hek_CNN1': load_model('../modeling/saved_models/hek-len50-Sp0.729-03-29 18:01.hdf5'),
    'hek_CNN2': load_model('../modeling/saved_models/hek-len50-Sp0.721-04-04 11:54.hdf5'),
    'pc3': load_model('../modeling/saved_models/pc3-len50-Sp0.785-04-07 18:08RNN.hdf5'),
    'muscle': load_model('../modeling/saved_models/muscle-len50-Sp0.874-04-07 19:58RNN.hdf5'),
    'egfp': load_model('../modeling/saved_models/retrained_main_MRL_model.hdf5'),
}

# model_dic = {
#     'egfp_1': load_model('../modeling/saved_models/Optimus_model1.hdf5'),
#     'egfp_2': load_model('../modeling/saved_models/Optimus_model2.hdf5'),
#     'egfp_3': load_model('../modeling/saved_models/Optimus_model3.hdf5'),
#     'egfp_4': load_model('../modeling/saved_models/Optimus_model4.hdf5'),
# }

for name in model_dic:
    print(name)
    scaler_key = str.split(name,'_')[0]
    scaler = scaler_dic[scaler_key]
    df[name] = scaler.inverse_transform(model_dic[name].predict(seq_e_test).reshape(-1))

df.to_csv('../data/RSV UTR Seeds Predicted.csv')



