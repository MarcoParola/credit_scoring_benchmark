#!/usr/bin/env python

"""
metrics.py: Implementation of utility functions for models metrics evaluation.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import numpy as np
from sklearn import metrics
from EMP.metrics import empCreditScoring

def gini_score(y_true, y_prob):
    """
    Computes the gini score (Somers'D).
    """
    assert (len(y_true) == len(y_prob))
    y_true_prob = np.asarray(np.c_[y_true, y_prob, np.arange(len(y_true))], dtype=float)
    y_true_prob = y_true_prob[np.lexsort((y_true_prob[:, 2], -1 * y_true_prob[:, 1]))]
    totalLosses = y_true_prob[:, 0].sum()
    giniSum = y_true_prob[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(y_true) + 1) / 2.
    return giniSum / len(y_true)

def normalized_gini_score(y_true, y_prob):
    """
    Computes the normalized gini score (Somers'D).
    """
    return gini_score(y_true, y_prob) / gini_score(y_true, y_true)

def brier_score(y_true, y_prob):
    """
    Computes the Brier Score using sklearn.metrics.
    """
    return metrics.brier_score_loss(y_true, y_prob)

def custom_brier_score(y_true, y_prob):
    """
    Computes the Brier Score using a custom implementation.
    """
    assert (len(y_true) == len(y_prob))
    losses = np.subtract(y_true, y_prob)**2
    brier_score = losses.sum()/len(y_true)
    return brier_score

def emp_score(y_true, y_prob):
    """
    Computes the expected maximum profit maximization score. It only returns the
    score.
    """
    assert (len(y_true) == len(y_prob))
    return empCreditScoring(y_prob, y_true, p_0=0.800503355704698, p_1=0.199496644295302, ROI=0.2644, print_output=False)[0]

def emp_score_frac(y_true, y_prob):
    """
    Computes the expected maximum profit maximization score. It returns both the
    score and the fraction of excluded.
    """
    assert (len(y_true) == len(y_prob))
    return empCreditScoring(y_prob, y_true, p_0=0.800503355704698, p_1=0.199496644295302, ROI=0.2644, print_output=False)

def accuracy_score(y_true, y_pred, classes):
    """
    Computes the accuracy using sklearn.metrics.accuracy_score().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.accuracy_score(y_true, y_pred)
    else:
        return metrics.accuracy_score(y_true, y_pred)

def accuracy_score(y_true, y_pred, classes):
    """
    Computes the accuracy using sklearn.metrics.accuracy_score().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.accuracy_score(y_true, y_pred)
    else:
        return metrics.accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred, classes):
    """
    Computes the f1 score using sklearn.metrics.f1_score().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.f1_score(y_true, y_pred, average='micro')
    else:
        return metrics.f1_score(y_true, y_pred, pos_label=1)

def precision_score(y_true, y_pred, classes):
    """
    Computes the precision score using sklearn.metrics.precision_score().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.precision_score(y_true, y_pred, average='micro')
    else:
        return metrics.precision_score(y_true, y_pred, pos_label=1)

def recall_score(y_true, y_pred, classes):
    """
    Computes the recall score using sklearn.metrics.recall_score().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.recall_score(y_true, y_pred, average='micro')
    else:
        return metrics.recall_score(y_true, y_pred, pos_label=1)

def cm_score(y_true, y_pred, classes):
    """
    Computes the confusion matrix score using sklearn.metrics.confusion_matrix().
    It handles binary and multi-class classification.
    """
    if len(classes) > 2:
        return metrics.confusion_matrix(y_true, np.argmax(y_pred, axis=-1))
    else:
        return metrics.confusion_matrix(y_true, y_pred)

def roc_curve(y_true, y_pred):
    """
    Computes the roc curve parameters score using sklearn.metrics.roc_curve().
    """
    return metrics.roc_curve(y_true, y_pred, pos_label=1)

def roc_auc_score(y_true, y_pred):
    """
    Computes the ROC AUC score using sklearn.metrics.roc_auc_score().
    """
    return metrics.roc_auc_score(y_true, y_pred)

def classification_report(y_true_train, y_pred_train, y_true_val, y_pred_val):
    """
    Prints final report for classifier model using sklearn.metrics.classification_report().
    """
    print(metrics.classification_report(y_true_train, y_pred_train, labels=[0,1]))
    print(metrics.classification_report(y_true_val, y_pred_val, labels=[0,1]))