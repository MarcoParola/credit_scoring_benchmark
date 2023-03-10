#!/usr/bin/env python

"""
utilities.py: Implementation of utility functions for plots.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def plot_hist(labels, values, title, xlabel, ylabel, figsize=(5,5),
              rotated_ticks=False, yticks=0, grid=False, save_path=None, dpi=100):
    """
    Plots the histogram for the given values using the provided labels.
    """
    value_counts_dict = {str(labels[i]):values[i] for i in range(len(labels))}

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')

    for key in value_counts_dict:
        bar = ax.bar(key, value_counts_dict[key], label=key)
        ax.bar_label(bar, labels=[value_counts_dict[key]])
        if rotated_ticks:
            ax.tick_params(labelrotation=90)

    fig.suptitle(title, fontsize=14, y=0.95)
    plt.xlabel(xlabel, fontsize=14, labelpad=10)
    plt.ylabel(ylabel, fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if yticks > 0:
        plt.yticks(np.arange(0, max(values)+1, yticks))

    if grid:
        ax.grid(axis='y', linestyle='dotted')

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()

def plot_dtypes_hist(data, title, xlabel, ylabel, figsize=(5,5),
                     rotated_ticks=False, save_path=None, dpi=100):
    """
    Plots the histogram for the data types found in the given dataframe.
    """
    features_types = pd.DataFrame(columns=['dtype', 'cout'])
    features_types['dtype'] = data.dtypes.value_counts().index.astype('string')
    features_types['cout'] = data.dtypes.value_counts().values
    features_types = features_types.groupby('dtype').sum()
    plot_hist(features_types.index, features_types['cout'].values, title, xlabel,
              ylabel, figsize, rotated_ticks, save_path=save_path, dpi=dpi)

def plot_numerical_boxplots(data, size=(12,6), save_path=None, dpi=100):
    """
    Plots the boxplot for all numerical features in the dataset.
    """
    boxplots_save_path = os.path.join(save_path, 'boxplots')
    if not os.path.isdir(boxplots_save_path):
        os.mkdir(boxplots_save_path)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=size, dpi=dpi, facecolor='w', edgecolor='k')
    fig.tight_layout(pad=5.0)
    ax = 0;

    for i, column in enumerate(data.select_dtypes(['float64', 'int64']).dtypes.index):
        axes[ax].boxplot(data[column])
        axes[ax].set_title(column, y=1.01)
        ax = (ax+1)%3
        if (ax == 0) and (i < len(data.select_dtypes(['float64', 'int64']).dtypes.index)-2):
            if save_path is not None:
                plt.savefig(boxplots_save_path+'/'+str(i)+'.pdf', format="pdf", bbox_inches="tight")
            fig, axes = plt.subplots(1, 3, figsize=size, dpi=dpi, facecolor='w', edgecolor='k')
            fig.tight_layout(pad=5.0)

    plt.show()

def plot_numerical_hist_kde(data, size=(12,6), save_path=None, dpi=100):
    """
    Plots both histogram and Kernel Density Estimation for all numerical features in the dataset.
    """
    hist_kde_save_path = os.path.join(save_path, 'hist_kde')
    if not os.path.isdir(hist_kde_save_path):
        os.mkdir(hist_kde_save_path)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=size, dpi=dpi, facecolor='w', edgecolor='k')
    fig.tight_layout(pad=5.0)
    ax = 0;

    for i, column in enumerate(data.select_dtypes(['float64', 'int64']).dtypes.index):
        data[column].plot(kind='hist', ax=axes[ax])
        data[column].plot(kind='kde', ax=axes[ax], secondary_y=True)
        axes[ax].set_title(column, y=1.01)
        axes[ax].tick_params(labelrotation=90)

        ax = (ax+1)%3
        if (ax == 0) and (i < len(data.select_dtypes(['float64', 'int64']).dtypes.index)-2):
            if save_path is not None:
                plt.savefig(hist_kde_save_path+'/'+str(i)+'.pdf', format="pdf", bbox_inches="tight")
            fig, axes = plt.subplots(1, 3, figsize=size, dpi=dpi, facecolor='w', edgecolor='k')
            fig.tight_layout(pad=5.0)

    plt.show()

def plot_features_scores(features, feature_scores, title, figsize=(10, 4), save_path=None, dpi=100):
    """
    Sorts and plots features scores.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_axes([0, 0, 1, 1])

    for score, f_name in sorted(zip(feature_scores, features), reverse=True)[:len(feature_scores)]:
        p = ax.bar(f_name, round(score, 2))
        ax.bar_label(p, label_type='edge', color='black', fontsize=16)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    fig.suptitle(title, fontsize=14, y=1.07)
    plt.xlabel('Features', fontsize=14, labelpad=15)
    plt.ylabel('Score', fontsize=14, labelpad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(['IV Score'], loc='best', prop={'size': 16})
    ax.set_ylim([0.0, 3.0])

    if save_path is not None:
        plt.savefig(save_path + 'features_scores.pdf', format="pdf", bbox_inches="tight")

    plt.show()

def plot_heatmap(data, figsize=(15, 15), save_path=None, dpi=100):
    """
    Plots the heatmap for the given input matrix using Red colors palette.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(data, annot=True, cmap=plt.cm.BuPu, fmt='.1f')

    if save_path is not None:
        plt.savefig(save_path + 'heatmap.pdf', format="pdf", bbox_inches="tight")

    plt.show()

def plot_pca_features(pca_features, target_feature):
    """
    Plots the 3 principal components given as input.
    The target feature is used to color samples based on the class label.
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 24), dpi=200)
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize=22, labelpad=20)
    ax.set_ylabel('Principal Component 2', fontsize=22, labelpad=20)
    ax.set_zlabel('Principal Component 3', fontsize=22, labelpad=20)

    targets = ['Defaulted', 'Non-Defaulted']
    colors = ['r', 'g']
    for target, color in zip([1.0, 0.0], colors):
        indicesToKeep = pca_features[target_feature]==target
        ax.scatter(pca_features.loc[indicesToKeep, 0],
                   pca_features.loc[indicesToKeep, 1],
                   pca_features.loc[indicesToKeep, 2],
                   c = color, marker=".")

    ax.legend(targets, fontsize=15, loc='upper left')
    ax.grid()

    plt.savefig('pca.pdf', format="pdf", bbox_inches="tight")

def plot_impurity_decrease(feature_importances, std):
    """
    The red bars are the feature importances of the forest, along with their
    inter-trees variability represented by the error bars.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
    feature_importances.plot.bar(yerr=std, ax=ax, color='r')

    fig.suptitle('Features importances using MDI', fontsize=15, y=0.95)
    plt.xlabel('Features', fontsize=12, labelpad=15)
    plt.ylabel('Mean decrease in impurity', fontsize=12, labelpad=15)

    plt.show()

def plot_train_loss_accuracy(train_loss, train_accuracy, val_loss, val_accuracy,
                             save_path=None, dpi=100):
    """
    Plots training and validation loss and acuracy over epochs.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(7, 7), dpi=dpi)
    plt.plot(train_loss,'g-', label="Training Set")
    plt.plot(val_loss,'r-', label="Validation Set")
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc="upper right")
    if save_path is not None:
        plt.savefig(save_path+'loss.pdf', format="pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 7), dpi=dpi)
    plt.plot(train_accuracy,'g-', label="Training Set")
    plt.plot(val_accuracy,'r-', label="Validation Set")
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path+'acc.pdf', format="pdf", bbox_inches="tight")
    plt.show()

    # print average values
    print("Average loss on train set: " + str(np.mean(train_loss)))
    print("Average accuracy on train set: " + str(np.mean(train_accuracy)))
    print("Average loss on test set: " + str(np.mean(val_loss)))
    print("Average accuracy on test set: " + str(np.mean(val_accuracy)))

def plot_accuracy(model_name, accuracies_train, accuracies_val, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(accuracies_train)+1), accuracies_train, '-o', range(1, len(accuracies_val)+1), accuracies_val, '-o')
    plt.xticks(range(1, len(accuracies_train)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Train Set', 'Validation Set'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'accuracy.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Train Set Accuracy: " + str(np.mean(accuracies_train)))
    print("Average Validation Set Accuracy: " + str(np.mean(accuracies_val)))

def plot_f1(model_name, f1_train, f1_val, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(f1_train)+1), f1_train, '-o', range(1, len(f1_val)+1), f1_val, '-o')
    plt.xticks(range(1, len(f1_train)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Train Set', 'Validation Set'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'f1.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Train Set F1 Score: " + str(np.mean(f1_train)))
    print("Average Validation Set F1 Score: " + str(np.mean(f1_val)))

def plot_precision(model_name, precision_train, precision_val, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(precision_train)+1), precision_train, '-o', range(1, len(precision_val)+1), precision_val, '-o')
    plt.xticks(range(1, len(precision_train)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Train Set', 'Validation Set'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'precision.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Train Set Precision: " + str(np.mean(precision_train)))
    print("Average Validation Set Precision: " + str(np.mean(precision_val)))

def plot_recall(model_name, recall_train, recall_val, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(recall_train)+1), recall_train, '-o', range(1, len(recall_val)+1), recall_val, '-o')
    plt.xticks(range(1, len(recall_train)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Train Set', 'Validation Set'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'recall.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Train Set Recall: " + str(np.mean(recall_train)))
    print("Average Validation Set Recall: " + str(np.mean(recall_val)))

def plot_gini(model_name, gini_scores, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(gini_scores)+1), gini_scores, '-o')
    plt.xticks(range(1, len(gini_scores)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Gini Coefficient'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'gini.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Gini Coefficient: " + str(np.mean(gini_scores)))

def plot_confusion_matrix(cnf_matrix, classes, normalize, title, save_path=None, dpi=100):
    """
    """
    plt.style.use("default")
    plt.figure(figsize=(6, 6), dpi=dpi, facecolor='w', edgecolor='k')

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('BuPu'))
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2%' if normalize else 'f'
    thresh = cnf_matrix.max() / 1.1

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path is not None:
        plt.savefig(save_path+'cm.pdf', format="pdf", bbox_inches="tight")

    plt.show()

def plot_roc_auc_score(y_test, y_pred, save_path=None, dpi=100):
    """
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    model_auc = auc(fpr, tpr)

    plt.style.use("ggplot")
    plt.figure(figsize=(7, 7), dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='(AUC = {:.3f})'.format(model_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('')
    plt.legend(loc='best')

    if save_path is not None:
        plt.savefig(save_path+'roc-auc.pdf', format="pdf", bbox_inches="tight")

    plt.show()

def plot_roc_auc_scores(model_name, fprs, tprs, thresholds, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(7, 7), dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot([0, 1], [0, 1], 'k--')

    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []

    for i in range(len(fprs)):
        model_auc = auc(fprs[i], tprs[i])
        plt.plot(fprs[i], tprs[i], label='Fold ' + str(i) + ' (AUC = {:.3f})'.format(model_auc), alpha=0.4)
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)

    mean_tpr = np.mean(tprs_interp, axis=0)
    model_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label='Mean ROC ' + str(i) + ' (AUC = {:.3f})'.format(model_auc), lw=2)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')

    if save_path is not None:
        plt.savefig(save_path+'roc-auc.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    print('Gini derived from mean AUC ' + str(2*model_auc-1))

def multiclass_roc_auc_score(y_test, y_pred, classes, average="micro", save_path=None, dpi=100):
    """
    Plots the ROC for multi-class classification.
    """
    # Binarize labels in a one-vs-all fashion
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    # ROC curve plot
    plt.style.use("ggplot")
    plt.figure(figsize=(7, 7), dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot([0, 1], [0, 1], 'k--')

    for (idx, c_label) in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx], y_pred[:,idx])
        plt.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

    # plot legend and axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curves per Class")
    plt.legend(loc='best')

    if save_path is not None:
        plt.savefig(save_path+'roc-auc.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    return roc_auc_score(y_test, y_pred, average=average)

def plot_brier(model_name, brier_scores, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(brier_scores)+1), brier_scores, '-o')
    plt.xticks(range(1, len(brier_scores)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['Brier Score'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'brier.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average Brier Score: " + str(np.mean(brier_scores)))

def plot_emp(model_name, emp_scores, emp_fractions, save_path=None, dpi=100):
    """
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi, facecolor='w', edgecolor='k')

    ax.plot(range(1, len(emp_scores)+1), emp_scores, '-o')
    ax.plot(range(1, len(emp_fractions)+1), emp_fractions, '-o')
    plt.xticks(range(1, len(emp_fractions)+1))
    fig.suptitle(model_name, fontsize=18, y=0.99)
    ax.legend(['EMPC', 'Fraction rejected'], loc='best', prop={'size': 10})

    if save_path is not None:
        plt.savefig(save_path + 'emp.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # print average values
    print("Average EMP: " + str(np.mean(emp_scores)))
    print("Average EMP Fractions: " + str(np.mean(emp_fractions)))
