import sys,os
from Bio import SeqIO
import pandas as pd
import Bio.SeqUtils.CodonUsage
import subprocess
from multiprocessing import Pool,cpu_count
from FeatureCommons import *
#import gzip
#from FeatureCommons import *
featList =list()
df = pd.read_csv('data/RSV UTR Seeds Predicted.csv')

feat2ID=dict()
i = 0
for seq in df['utr']:
    #featList.append(foldenergy_feature(seq))
    ##DNA CG composition
    featList = list(singleNucleotide_composition(seq).items())
    ##RNA folding
    featList += list(foldenergy_feature(seq).items())
    for featItem in featList:
        featname = featItem[0]
        featVal = featItem[1]
        df.loc[i, featname] = featVal
    print(i, end = ' ')
    print(seq)
    i += 1


df.to_csv('data/RSV UTR Seeds Predicted_features.csv')




print("done")