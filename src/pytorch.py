#!/usr/bin/env python

"""
pytorch.py: Implementation of utility functions for PyTorch models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import numpy as np

from torch import Tensor
from torch.nn import Module, Linear, ReLU, Sigmoid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.init import kaiming_uniform_, xavier_uniform_

from sklearn.preprocessing import LabelEncoder

class CSVDataset(Dataset):
    def __init__(self, df, target):
        self.X = np.array(df.drop([target], axis=1, inplace=False))
        self.X = self.X.astype('float32')

        self.y = np.array(df[target])
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def create_dl(data, batch_size, shuffle):
    """
    Creates a PyTorch DataLoader for the given dataset.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

class MLP(Module):
    """
    Multilayer Perceptron.
    """
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)

        return X