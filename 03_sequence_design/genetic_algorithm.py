import random
import math

import pandas as pd
import numpy as np


def ret_rand_nuc():
    x = random.randint(0, 3)
    if x == 0:
        return [1, 0, 0, 0]  # A
    if x == 1:
        return [0, 1, 0, 0]  # C
    if x == 2:
        return [0, 0, 1, 0]  # G
    if x == 3:
        return [0, 0, 0, 1]  # T


def vector_to_nuc(arr, seq_len=50):
    seq = ''
    for i in range(seq_len):
        if arr[i, 0] == 1:
            seq = seq + 'A'
        if arr[i, 1] == 1:
            seq = seq + 'C'
        if arr[i, 2] == 1:
            seq = seq + 'G'
        if arr[i, 3] == 1:
            seq = seq + 'T'
    return seq


def convert_and_save(sequences, predictions):
    # Convert the one-hot encoded sequences to A, C, T, G
    seqs = []
    for nbr in range(len(sequences)):
        seqs.append(vector_to_nuc(sequences[nbr]))
    df = pd.DataFrame(data=[seqs, predictions.tolist()]).transpose()
    df.columns = ['utr', 'prediction']
    df.sort_values('prediction', ascending=False, inplace=True)
    return df


def make_random_sequences(nbr_sequences, length, constant='', no_uaug=False, no_stop=False):
    # Make randomize sequences, allowing for the inclusion / exclusion of uATGs / stop codons
    seqs = []
    nucs = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    i = 0
    while i < nbr_sequences:
        new_seq = ''
        for n in range(length - len(constant)):
            new_seq = new_seq + nucs[random.randint(0, 3)]

        if no_uaug == False or (no_uaug == True and 'ATG' not in new_seq):
            if no_stop == False or (
                    no_stop == True and ('TAG' not in new_seq and 'TGA' not in new_seq and 'TAA' not in new_seq)):
                new_seq = new_seq + constant
                seqs.append(new_seq)
                i += 1
    return seqs


def simple_mutate(seq, nbr_bases=1, prob=1):
    if nbr_bases > 1 and prob > random.random():
        nbr_bases = nbr_bases
    else:
        nbr_bases = 1
    for i in range(nbr_bases):
        pos = random.randint(0, len(seq)-1)
        #if 'N', keep it no changed
        if (seq[pos]== [0, 0, 0, 0]).all():
            continue
        seq[pos] = ret_rand_nuc()

    return seq


def check_for_uaug(seq):
    seq = vector_to_nuc(seq)
    return 'ATG' in seq[:50]


def check_for_stops(seq):
    seq = vector_to_nuc(seq)
    if 'TAG' in seq[:50] or 'TGA' in seq[:50] or 'TAA' in seq[:50]:
        return True
    return False


def negative_selection(seq, model, scaler, target_val, no_uaug=False, no_stop=False, nbr_bases_to_mutate=1,
                       multi_mutate_prob=1, seq_len=50):
    seqs = np.empty([2, seq_len, 4])
    seqs[0] = seq.copy()
    seqs[1] = simple_mutate(seq.copy(), nbr_bases=nbr_bases_to_mutate, prob=multi_mutate_prob)

    if no_uaug == True and check_for_uaug(seqs[1]):
        return seqs[0]
    if no_stop == True and check_for_stops(seqs[1]):
        return seqs[0]

    scores = model.predict(seqs).reshape(-1)
    scores = scaler.inverse_transform(scores)
    if scores[1] < scores[0]:
        if scores[1] >= target_val:
            return seqs[1]
        else:
            return seqs[0]
    else:
        return seqs[0]


def selection(seq, prefix, model, scaler, target_val, no_uaug=False, no_stop=False, nbr_bases_to_mutate=1, multi_mutate_prob=1,
              seq_len=50):
    seqs = np.empty([2, seq_len, 4])
    seqs[0] = seq.copy()
    seq_mutated = simple_mutate(seq[prefix:, 0:].copy(), nbr_bases=nbr_bases_to_mutate, prob=multi_mutate_prob)
    prefix_seq = seq[0:prefix, 0:].copy()
    seqs[1] = np.concatenate((prefix_seq, seq_mutated))

    if no_uaug == True and check_for_uaug(seqs[1]):
        return seqs[0]
    if no_stop == True and check_for_stops(seqs[1]):
        return seqs[0]

    scores = model.predict(seqs).reshape(-1)
    scores = scaler.inverse_transform(scores)
    if scores[1] > scores[0]:
        if scores[1] <= target_val:
            return seqs[1]
        else:
            return seqs[0]
    else:
        return seqs[0]


def selection_to_target(seq, model, scaler, target_val, no_uaug=False, no_stop=False, nbr_bases_to_mutate=1,
                        multi_mutate_prob=1, seq_len=50, accept_range=0.1):
    seqs = np.empty([2, seq_len, 4])
    # Save the incoming sequence before mutating
    seqs[0] = seq.copy()
    # The mutated sequence
    seqs[1] = simple_mutate(seq.copy(), nbr_bases=nbr_bases_to_mutate, prob=multi_mutate_prob)

    # Decide whether to continue with the new sequence based on the uAUG / stop codon preference
    if no_uaug == True and check_for_uaug(seqs[1]):
        return seqs[0]
    if no_stop == True and check_for_stops(seqs[1]):
        return seqs[0]

    scores = model.predict(seqs).reshape(-1)
    scores = scaler.inverse_transform(scores)

    # Accept sequences that fall within this range. May provide more sequence diversity
    if scores[0] >= target_val - accept_range and scores[0] <= target_val + accept_range:
        return seqs[0]
    else:
        if abs(target_val - scores[1]) <= abs(target_val - scores[0]):
            return seqs[1]
        else:
            return seqs[0]


def selection_combo(seq, prefix, model1, model2, scaler1, scaler2, target_val1, target_val2, no_uaug=False, no_stop=False,
                    nbr_bases_to_mutate=1, multi_mutate_prob=1, seq_len=50):
    seqs = np.empty([2, seq_len, 4])
    seqs[0] = seq.copy()
    seq_mutated = simple_mutate(seq[prefix:, 0:].copy(), nbr_bases=nbr_bases_to_mutate, prob=multi_mutate_prob)
    prefix_seq = seq[0:prefix, 0:].copy()
    seqs[1] = np.concatenate((prefix_seq, seq_mutated))
    if no_uaug == True and check_for_uaug(seqs[1]):
        return seqs[0]
    if no_stop == True and check_for_stops(seqs[1]):
        return seqs[0]

    scores1 = model1.predict(seqs).reshape(-1)
    scores1 = scaler1.inverse_transform(scores1)
    scores2 = model2.predict(seqs).reshape(-1)
    scores2 = scaler2.inverse_transform(scores2)
    if (scores1[1] > scores1[0]) & (scores2[1] > scores2[0]):
        if (scores1[1] <= target_val1) | (scores2[1] <= target_val2):
            return seqs[1]
        else:
            return seqs[0]
    else:
        return seqs[0]
