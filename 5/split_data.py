import pandas as pd
import numpy as np

"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

DATA_DIR = 'data/data.tsv'
TRAIN = 0.64
VALIDATION = 0.16
TEST = 0.2
OVERFIT_SIZE = 50

raw_data = pd.read_csv(DATA_DIR, sep='\t')


def split_data(data):

    objective = data[data['label'] == 0]
    subjective = data[data['label'] == 1]

    # Find indices to split data
    first_split = int(np.floor(TRAIN * len(objective)))
    second_split = int(np.floor(VALIDATION * len(objective))) + first_split

    # Split data and shuffle
    train_data = pd.concat([subjective[:first_split], objective[:first_split]], ignore_index=True).sample(frac=1)\
        .reset_index(drop=True)
    validation_data = pd.concat([subjective[first_split:second_split], objective[first_split:second_split]],
                                ignore_index=True).sample(frac=1).reset_index(drop=True)
    print(validation_data)
    test_data = pd.concat([subjective[second_split:], objective[second_split:]], ignore_index=True).sample(frac=1)\
        .reset_index(drop=True)
    overfit_data = pd.concat([subjective[:OVERFIT_SIZE], objective[:OVERFIT_SIZE]], ignore_index=True).sample(frac=1)\
        .reset_index(drop=True)

    # Write to files
    train_data.to_csv('data/train.tsv', sep='\t', index=False)
    validation_data.to_csv('data/validation.tsv', sep='\t', index=False)
    test_data.to_csv('data/test.tsv', sep='\t', index=False)
    overfit_data.to_csv('data/overfit.tsv', sep='\t', index=False)

    print('Count of objective and subjective data')
    print('Training:')
    print(train_data['label'].value_counts())
    print('Validation:')
    print(validation_data['label'].value_counts())
    print('Test:')
    print(test_data['label'].value_counts())


split_data(raw_data)
