import pandas as pd
import numpy as np
#df = pd.read_csv('../data/GSM3130435_egfp_unmod_1.csv')
#df = pd.read_csv('../data/Hek_R200bp5UTR_RNA5Ribo0.1.csv')
#df.sort_values('total_reads', ascending=False).reset_index(drop=True)

# Select a number of UTRs for the purpose of scaling.


#print(df.describe())
#df['rl'] = df['rl'].apply(np.log2)

#print(df.describe())


#df = pd.read_csv('../data/GSM3130443_designed_library.csv')
# df = pd.read_csv('../sequence_design/Evolved_UTRs_50bp_Hek.csv')
#
# print(df.describe())
# human = df[(df['library'] == 'human_utrs') | (df['library'] == 'snv')]
# human = human[human['designed'] == True]
# human = human.loc[:, 'rl']
# print(human.describe())
import keras
model = keras.models.load_model('./saved_models/hek-len50-Sp0.675-04-07 00:31.hdf5')
model.summary()

