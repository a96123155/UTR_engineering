import pandas as pd
import numpy as np
from pandas import Series

data_file = '../data/RNA protein Liver UTRs.csv'
df = pd.read_csv(data_file)
utr5 = df.loc[df['Seq Type'] ==  '5UTR']
utr3 = df.loc[df['Seq Type'] ==  '3UTR']
out_df = pd.DataFrame(columns=['Description', '5UTR','3UTR'])
for i,utr5_row in utr5.iterrows():
    for j,utr3_row in utr3.iterrows():
        seq_name = utr5_row['Description']+'-5UTR + '+utr3_row['Description']+'-3UTR'
        new_row = {'Description':seq_name, '5UTR':utr5_row['Sequence'], '3UTR':utr3_row['Sequence']}
        out_df = out_df.append(new_row, ignore_index=True)


out_df.to_csv('../data/RNA protein Liver UTRs Combined.csv')
