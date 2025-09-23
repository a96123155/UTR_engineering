import pandas as pd
import numpy as np
import keras
np.random.seed(1337)
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from genetic_algorithm import *
import csv


pd.set_option("display.max_colwidth",100)
sns.set(style="ticks", color_codes=True)
def plot_data(data, x, y, x_title='x', y_title='y', xlim=None, ylim=None, size=4, alpha=0.02):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x],data[y])
    print('# of UTRs = ', len(data))
    print('r-squared = ',r_value**2)

    sns.set(style="ticks", color_codes=True)
    g = sns.JointGrid(data=data, x=x, y=y, xlim=xlim, ylim=ylim, size=size)
    g = g.plot_joint(plt.scatter, color='#e01145', edgecolor="black", alpha=alpha)
    f = g.fig
    f.text(x=0, y=0, s='r2 = {}'.format(round(r_value**2, 3)))
    g = g.plot_marginals(sns.distplot, kde=False, color='#e01145')
    g = g.set_axis_labels(x_title, y_title)

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}
    return np.array([ltrdict[x] for x in seq])
#
# from keras.preprocessing import sequence
# from tensorflow.keras.optimizers import RMSprop
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.layers.core import Dropout
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.constraints import maxnorm
# from keras.callbacks import ModelCheckpoint, EarlyStopping


def test_data(df, model, test_seq, obs_col, output_col='pred'):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].reshape(-1, 1))
    # df.loc[:,'obs_stab'] = test_df['stab_df']
    predictions = model.predict(test_seq).reshape(-1)
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def binarize_sequences(df, col='utr', seq_len=54):
    vector = np.empty([len(df), seq_len, 4])
    for i, seq in enumerate(df[col].str[:seq_len]):
        vector[i] = vectorizeSequence(seq.lower())
    return vector


df = pd.read_csv('../data/Hek_R200bp5UTR_RNA5Ribo0.1.csv')
#df.sort_values('total_reads', ascending=False).reset_index(drop=True)
df['rl'] = df['rl'].apply(np.log2)
# Scale
scaler1 = preprocessing.StandardScaler()
scaler1.fit(df['rl'].values.reshape(-1,1))

df = pd.read_csv('../data/GSM3130435_egfp_unmod_1.csv')
df.sort_values('total_reads', ascending=False).reset_index(drop=True)
scale_utrs = df[:40000]
# Scale
scaler2 = preprocessing.StandardScaler()
scaler2.fit(df['rl'].values.reshape(-1,1))

from keras.models import load_model
# model = load_model('../modeling/saved_models/evolution_model.hdf5')
# model = load_model('../modeling/saved_models/retrained_evolution_model.hdf5')
# model = load_model('../modeling/saved_models/main_MRL_model.hdf5')
#model = load_model('../modeling/saved_models/retrained_main_MRL_model.hdf5')
model1 = load_model('../modeling/saved_models/hek-len50-Sp0.745-04-07 22:50RNN.hdf5')
model2 = load_model('../modeling/saved_models/retrained_main_MRL_model.hdf5')



# Dictionary where new sequences are saved
evolved_seqs = {}
# Number of evolution iterations
iterations = 150
# Number of bases to mutate if the probability to 'multi-mutate' is exceeded
nbr_bases_to_mutate = 2
# Probability to change multiple bases in an iteration
prob_of_multi_mutation = 0.5
# If using the original evolution model, set seq_len to 54. That model was
# trained on UTRs that included the first for basees of the CDS (ATGG).
seq_len = 50
prefix_seq = 'AG'
# Choose whether or not to allow uAUGs and / or stop codons
no_uaug = True
no_stop = False
# Evolve to highest MRL - set target_rl to arbitrarily high value
target_rl_1 = 20
# Highest log2RL
target_rl_2 = 10

nbr_sequences = 500
rand_seqs = make_random_sequences(nbr_sequences, seq_len-len(prefix_seq), no_uaug=True, no_stop=True)
e_seqs = np.empty([len(rand_seqs), seq_len, 4])

result_df = pd.DataFrame(columns=('UTR','Predicted log2TE','Predicted_MRL'))
best_pred_file_name = 'Evolved_UTRs_50bp_Combo_prefix0420.csv'
iterations_pred_file_name = 'Evolved_UTRs_50bp_Combo_prefix_iterations0420.csv'
# vectorizeSequence
i = 0
for seq in rand_seqs:
    seq = prefix_seq + seq
    e_seqs[i] = vectorizeSequence(seq.lower())
    i += 1
# assign an empty array for evolved seqs
for x in range(nbr_sequences):
    evolved_seqs[x] = np.empty((iterations, 3), dtype=object)
# save the better seqs and predicted value to the evolved_seqs
for gen in range(0, iterations):
    for i in range(len(e_seqs)):
        e_seqs[i] = selection_combo(seq=e_seqs[i], prefix=2, model1=model1, model2=model2, scaler1=scaler1, scaler2=scaler2,
                                    target_val1=target_rl_1, target_val2=target_rl_2,no_uaug=no_uaug,
                                    no_stop=no_stop, nbr_bases_to_mutate=nbr_bases_to_mutate,
                                    multi_mutate_prob=prob_of_multi_mutation, seq_len=seq_len)
        if i % 50 == 0:
            print(i, end=' ')

    for x in range(nbr_sequences):
        evolved_seqs[x][gen, 0] = vector_to_nuc(e_seqs[x])
        evolved_seqs[x][gen, 1] = model1.predict(e_seqs).reshape(-1)[x]
        evolved_seqs[x][gen, 2] = model2.predict(e_seqs).reshape(-1)[x]

    if gen % 5 == 0:
        print('Generation ' + str(gen))

# save to data frame and file
for x in range(nbr_sequences):
    evolved_seqs[x][:, 1] = scaler1.inverse_transform(evolved_seqs[x][:, 1])
    evolved_seqs[x][:, 2] = scaler2.inverse_transform(evolved_seqs[x][:, 2])
    #print(evolved_seqs[x][-1:])
    result_df.loc[x+1] = [evolved_seqs[x][-1, 0], evolved_seqs[x][-1, 1], evolved_seqs[x][-1, 2]]

result_df.to_csv(best_pred_file_name)

# save the all sequence in  iterations to csv

with open(iterations_pred_file_name, 'w', newline='') as f:
    fieldnames = ['No', 'iterations','Seq',' Prediction log2TE', 'Prediction MRL']
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for x in range(nbr_sequences):
        for i in range(iterations):
            writer.writerow([x, i, evolved_seqs[x][i, 0], evolved_seqs[x][i, 1], evolved_seqs[x][i, 2]])



