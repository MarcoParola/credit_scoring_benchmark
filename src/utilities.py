#!/usr/bin/env python

"""
utilities.py: Implementation of general purpose utility functions.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv(dataset_path, dtype=None, sep=',', header='infer'):
    """
    Loads the provided .csv file using pandas.
    """
    return pd.read_csv(dataset_path, dtype=dtype, sep=sep, header=header)

def read_excel(dataset_path, dtype=None, header=0):
    """
    Loads the provided .xls file using pandas.
    """
    return pd.read_excel(dataset_path, dtype=dtype, header=header)

def create_directory(dir_path):
    """
    Creates the specified directory.
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def int64_to_float64(data):
    """
    Converts all int64 columns of the dataframe to float64.
    """
    for column in data.dtypes.index:
        if data.dtypes[column] in ['int64']:
            data[column] = data[column].astype('float64')

def object_to_category(data):
    """
    Converts all object columns of the dataframe to category.
    """
    for column in data.dtypes.index:
        if data.dtypes[column] in ['object']:
            data[column] = data[column].astype('category')

def object_to_float64(data):
    """
    Converts all object columns of the dataframe to float64.
    """
    for column in data.dtypes.index:
        if data.dtypes[column] in ['object']:
            data[column] = data[column].astype('float64')

def replace_to_nan(data, word):
    """
    Replaces the given word with np.nan.
    """
    data.replace(word, np.nan, inplace=True)

def save_dataset(data, features_scores, test_size, save_path):
    """
    Splits the preprocessed dataset in train and test and saves the dataframes
    using the .parquet file format. The features scores dictionary is saved as well.
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # store features selection scores
    with open(save_path + '/features_scores.json', 'w') as outfile:
        json.dump(features_scores, outfile)

    train_split, test_split = train_test_split(data, test_size=test_size, shuffle=True, stratify=data[['defaulted']])
    train_split.to_parquet(save_path + '/train.parquet', index=False)
    test_split.to_parquet(save_path + '/test.parquet', index=False)

    print("Train split size: " + str(len(train_split)))
    print("Test split size: " + str(len(test_split)))