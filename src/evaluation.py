#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for evaluation models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os

from tqdm import tqdm

from src import pytorch

from numpy import vstack

from torch.optim import SGD
from torch.nn import BCELoss

from datetime import datetime

from sklearn.metrics import accuracy_score

def is_dl_model(model):
    """
    Checks if the given model is based on PyTorch.
    """
    return 'pytorch' in model.__class__.__module__

def k_fold_cross_validate(model, k, train, test):
    """
    """
    # model train/test results save directory
    save_path = '../../../models/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    print('Model save_path: ' + save_path)

    if is_dl_model(model):
        train_data = pytorch.CSVDataset(train, 'defaulted')
        test_data = pytorch.CSVDataset(test, 'defaulted')
        train_dl = pytorch.create_dl(train_data, batch_size=32, shuffle=True)
        test_dl = pytorch.create_dl(test_data, batch_size=1024, shuffle=False)

        train_dl_model(train_dl, model)
        evaluate_dl_model(test_dl, model)

def train_dl_model(train_dl, model):
    """
    Trains a deep learning model.
    Based on PyTorch.
    """
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in tqdm(range(100)):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_dl_model(test_dl, model):
    """
    Evaluates a deep learning model.
    Based on PyTorch.
    """
    y_pred, y_true = list(), list()

    for i, (inputs, targets) in enumerate(test_dl):
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        y_true.append(actual)

        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        yhat = yhat.round()
        y_pred.append(yhat)

    y_pred, y_true = vstack(y_pred), vstack(y_true)

    acc = accuracy_score(y_true, y_pred)

    print('Accuracy: %.3f' % acc)