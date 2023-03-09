#!/usr/bin/env python

"""
metrics.py: Implementation of utility functions for models metrics evaluation.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

from src import plotting

import numpy as np
import pandas as pd
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

def confusion_matrix_report(model_name, conf_matrix_list, classes, save_path=None, dpi=100):
    """
    Computes average normalized confusion matrix and plots it.
    """
    mean_of_conf_matrix_list = np.mean(conf_matrix_list, axis=0)

    plotting.plot_confusion_matrix(cnf_matrix=mean_of_conf_matrix_list,
                                   classes=classes, normalize=True,
                                   title='Normalized confusion matrix',
                                   save_path=save_path, dpi=dpi)

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

def final_report(performance_metrics, save_path, model_name, classes):
    """
    Stores the given performance metrics as a .csv files and plots all
    performance metrics.
    """
    # store metrics as csv file
    metrics_dict = {'accuracy': performance_metrics['val_accuracy'],
                    'f1': performance_metrics['val_f1'],
                    'precision': performance_metrics['val_precision'],
                    'recall': performance_metrics['val_recall'],
                    'auc': performance_metrics['auc'],
                    'gini': performance_metrics['gini'],
                    'brier': performance_metrics['brier'],
                    'emp': performance_metrics['emp_score'],
                    'emp_frac': performance_metrics['emp_frac']}
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(save_path + '/train_metrics.csv', index=False)

    plot_save_path = save_path + '/' + model_name + '-'

    # accuracy report and plot
    plotting.plot_accuracy(model_name = '',
                           accuracies_train = performance_metrics['train_accuracy'],
                           accuracies_val = performance_metrics['val_accuracy'],
                           save_path = plot_save_path,
                           dpi = 100)

    # f1 score and plot
    plotting.plot_f1(model_name = '',
                     f1_train = performance_metrics['train_f1'],
                     f1_val = performance_metrics['val_f1'],
                     save_path = plot_save_path,
                     dpi = 100)
    
    # precision score and plot
    plotting.plot_precision(model_name = '',
                    precision_train = performance_metrics['train_precision'],
                    precision_val = performance_metrics['val_precision'],
                    save_path = plot_save_path,
                    dpi = 100)

    # recall score and plot
    plotting.plot_recall(model_name = '',
                         recall_train = performance_metrics['train_recall'],
                         recall_val = performance_metrics['val_recall'],
                         save_path = plot_save_path,
                         dpi = 100)

    # confusion matrix
    confusion_matrix_report(model_name = '',
                            conf_matrix_list = performance_metrics['conf_matrix'],
                            classes = classes,
                            save_path = plot_save_path,
                            dpi = 100)

    # roc curves and auc scores
    plotting.plot_roc_auc_scores(model_name = '',
                                 fprs = performance_metrics['fpr'],
                                 tprs = performance_metrics['tpr'],
                                 thresholds = performance_metrics['thresh'],
                                 save_path = plot_save_path,
                                 dpi = 100)
    
    # plot folds gini scores
    plotting.plot_gini(model_name = '',
                       gini_scores = performance_metrics['gini'],
                       save_path = plot_save_path,
                       dpi = 100)

    # plot folds brier scores
    plotting.plot_brier(model_name = '',
                        brier_scores = performance_metrics['brier'],
                        save_path = plot_save_path,
                        dpi = 100)

    # plot folds error costs
    plotting.plot_emp(model_name = '',
                      emp_scores = performance_metrics['emp_score'],
                      emp_fractions = performance_metrics['emp_frac'],
                      save_path = plot_save_path,
                      dpi = 100)