#!/usr/bin/env python

"""
utilities.py: Implementation of utility functions for plots.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    ax.set_ylim([0.0, 1.0])

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