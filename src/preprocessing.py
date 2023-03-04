#!/usr/bin/env python

"""
preprocessing.py: Implementation of utility functions for preprocessing.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from mdlp.discretization import MDLP
from optbinning import OptimalBinning
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectKBest, chi2, f_classif

laplacian_smoothing = 0.0000000000000000000000001

def one_hot_encode(data, feature):
    """
    Applies one-hot-encoding for the given feature.
    Returns the provided dataset merged with the one-hot-encoded feature.
    """

    ohe = pd.get_dummies(data[feature], prefix=feature, prefix_sep='-').astype('float64')
    return data.merge(ohe, left_index=True, right_index=True)

def woe_encode(data, woe_feature, target_feature, normalize):
    """
    Applies WoE-encoding for the given feature.
    Returns the provided dataset merged with the WoE-encoded feature.
    """
    if normalize:
        woe_crosstab = (pd.crosstab(data[woe_feature], data[target_feature], normalize='columns')+laplacian_smoothing).assign(woe=lambda dfx: np.log(dfx[True] / dfx[False]))
    else:
        woe_crosstab = (pd.crosstab(data[woe_feature], data[target_feature])+laplacian_smoothing).assign(woe=lambda dfx: np.log(dfx[True] / dfx[False]))

    woe_mappings = dict(zip(woe_crosstab[False].index.categories, woe_crosstab['woe']))
    data[woe_feature+'-woe'] = data[woe_feature].map(woe_mappings).astype('float64')
    return data

def iterative_imputer(data, iterations):
    """
    Categorical features are first dropped because the fit() method of IterativeImputer
    only works with numeric features. Categorical features are added back before
    returning the new dataframe with missing values replaced with imputed values.
    """
    # store column types
    dtypes = data.dtypes

    # drop categorical features
    dropped_features = []
    for column in data.dtypes.index:
        if str(data.dtypes[column]) == 'category':
            dropped_features.append(data[[column]])
            data.drop([column], axis=1, inplace=True)

    # run iterative imputer
    iterative_imputer = IterativeImputer(max_iter=iterations)
    iterative_imputer.fit(data)
    data = pd.DataFrame(iterative_imputer.transform(data), columns=data.columns)

    # restore dropped columns
    for column in dropped_features:
        data = data.merge(column, left_index=True, right_index=True)

    # restore column types
    data = data.astype(dtypes)

    return data

def opt_bin_woe(data, solver, outlier_detector, save_path, verbose):
    """
    Applies WoE encoding after binning using OptimalBinning to the given
    dataframe features and computes IV scores.

    :param data: pandas.DataFrame to be preprocessed.
    :param solver: OptimalBinning solver to be used (cp, mip).
    :param outlier_detector: OptimalBinning outlier_detector to be used (range, zscore).
    :param verbose: set to True for verbose logging.

    :return: preprocessed pandas.DataFrame and features scores dictionary.
    """ 
    features_scores = {}
    column_dtype = ''

    optbinning_save_path = os.path.join(save_path, 'optbinning')
    if not os.path.isdir(optbinning_save_path):
        os.mkdir(optbinning_save_path)

    for column in tqdm(data.dtypes.index):
        if verbose:
            print('Processing feature: ' + column + '.')

        if data.dtypes[column] in ['float64', 'int64']:
            column_dtype = "numerical"
        elif data.dtypes[column] in ['category']:
            column_dtype = "categorical"
        elif data.dtypes[column] in ['bool']:
            continue

        x = data[column].values
        y = data.defaulted
        optb = OptimalBinning(name=column, dtype=column_dtype, solver=solver,
                              outlier_detector=outlier_detector, verbose=verbose)
        optb.fit(x, y)
        binning_table = optb.binning_table
        binning_table_df = binning_table.build()
        features_scores[column] = binning_table_df['IV']['Totals']
        binning_table.plot(metric="woe", show_bin_labels=True,
                           savefig=optbinning_save_path+'/'+column+'-binning.pdf',
                           figsize=(10,10))
        data[column] = optb.transform(x, metric="woe")
        
        if verbose:
            print('Solver status for feature ' + column + ': ' + optb.status)
            print(binning_table_df['Bin'])
            print(pd.Series(data[column]).value_counts())

    return data, features_scores

def features_correlation(data):
    """
    Computes features correlation using both Pearson and Spearman coefficients.
    """
    pearson_cor = data.corr(method='pearson')
    spearman_cor = data.corr(method='spearman')
    cor = (pearson_cor+spearman_cor)/2

    return cor

def isolation_forest_outliers(data, target_feature, iterations, splits, estimators, contamination, jobs):
    """
    Detects outliers using the isolation forest algorithm applied using repeated K-fold cross validation.
    It returns the dataframe containing the outliers detected more than once is.
    """
    X = np.array(data.select_dtypes(include=['float64', 'int64']))
    y = np.array(data[target_feature])
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    outliers = pd.DataFrame(columns=['ID', 'predict', 'anomaly_score'])

    for i in tqdm(range(iterations)):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            isf = IsolationForest(n_estimators=estimators, n_jobs=jobs, contamination=contamination)
            isf.fit(X_train)
            isf_outliers = pd.DataFrame(isf.predict(X_test), columns=['predict'])
            isf_anomaly_scores = pd.DataFrame(isf.score_samples(X_test), columns=['anomaly_score'])
            isf_outliers = isf_outliers.merge(isf_anomaly_scores, left_index=True, right_index=True)
            isf_outliers = isf_outliers[isf_outliers['predict']==-1]

            if len(isf_outliers) > 0:
                for index, row in isf_outliers.iterrows():
                    new_row = pd.DataFrame({'ID':test_index[index], 'predict':row['predict'],
                                            'anomaly_score':row['anomaly_score']}, index=[0])
                    outliers = pd.concat([outliers, new_row])

    outliers = outliers.groupby('ID').sum()
    return outliers[outliers['predict'] < -1]

def local_outlier_factor(data, target_feature, iterations, splits, algo, jobs, contamination):
    """
    Detects outliers using the local outlier factor algorithm applied using repeated K-fold cross validation.
    It returns the dataframe containing the outliers detected more than once is.
    """
    X = np.array(data.select_dtypes(include=['float64', 'int64']))
    y = np.array(data[target_feature])
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    outliers = pd.DataFrame(columns=['ID', 'predict', 'anomaly_score'])

    for i in tqdm(range(iterations)):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            lof = LocalOutlierFactor(n_neighbors=100, novelty=True, algorithm=algo, n_jobs=jobs, contamination=contamination)
            lof.fit(X_train)
            lof_outliers = pd.DataFrame(lof.predict(X_test), columns=['predict'])
            lof_anomaly_scores = pd.DataFrame(lof.score_samples(X_test), columns=['anomaly_score'])
            lof_outliers = lof_outliers.merge(lof_anomaly_scores, left_index=True, right_index=True)
            lof_outliers = lof_outliers[lof_outliers['predict']==-1]

            if len(lof_outliers) > 0:
                for index, row in lof_outliers.iterrows():
                    new_row = pd.DataFrame({'ID':test_index[index], 'predict':row['predict'],
                                            'anomaly_score':row['anomaly_score']}, index=[0])
                    outliers = pd.concat([outliers, new_row])

    outliers = outliers.groupby('ID').sum()
    return outliers[outliers['predict'] < -1]

def one_class_svm_outliers(data, target_feature, iterations, splits, nu, kernel, gamma):
    """
    Detects outliers using the local outlier factor algorithm applied using repeated K-fold cross validation.
    It returns the dataframe containing the outliers detected more than once is.
    """
    X = np.array(data.select_dtypes(include=['float64', 'int64']))
    y = np.array(data[target_feature])
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    outliers = pd.DataFrame(columns=['ID', 'predict', 'anomaly_score'])

    for i in tqdm(range(iterations)):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            ocs = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, max_iter=10000)
            ocs.fit(X_train)
            ocs_outliers = pd.DataFrame(ocs.predict(X_test), columns=['predict'])
            ocs_anomaly_scores = pd.DataFrame(ocs.score_samples(X_test), columns=['anomaly_score'])
            ocs_outliers = ocs_outliers.merge(ocs_anomaly_scores, left_index=True, right_index=True)
            ocs_outliers = ocs_outliers[ocs_outliers['predict']==-1]

            if len(ocs_outliers) > 0:
                for index, row in ocs_outliers.iterrows():
                    new_row = pd.DataFrame({'ID':test_index[index], 'predict':row['predict'],
                                            'anomaly_score':row['anomaly_score']}, index=[0])
                    outliers = pd.concat([outliers, new_row])

    outliers = outliers.groupby('ID').sum()
    return outliers[outliers['predict'] < -1]

def min_max_normalization(data):
    """
    Applies column-wise min-max normalization to the input pandas DataFrame.
    Categorical, boolean, integer and woe-encoded features are not normalized.
    """
    # store column types
    dtypes = data.dtypes

    # drop categorical features
    dropped_features = []
    for column in data.dtypes.index:
        if (str(data.dtypes[column]) == 'category' or
            str(data.dtypes[column]) == 'bool' or
            str(data.dtypes[column]) == 'int64' or
            "woe" in str(column) or
            (data[column] == 0).all()):
            dropped_features.append(data[[column]])
            data.drop([column], axis=1, inplace=True)

    data = (data-data.min())/(data.max()-data.min())
    
    # restore dropped columns
    for column in dropped_features:
        data = data.merge(column, left_index=True, right_index=True)

    # restore column types
    data = data.astype(dtypes)

    return data

def z_score_normalization(data):
    """
    Applies column-wise z-score normalization to the dataset.
    Categorical, boolean, integer and woe-encoded features are not normalized.
    """
    # store column types
    dtypes = data.dtypes

    # drop categorical features
    dropped_features = []
    for column in data.dtypes.index:
        if (str(data.dtypes[column]) == 'category' or
            str(data.dtypes[column]) == 'bool' or
            str(data.dtypes[column]) == 'int64' or
            "woe" in str(column) or
            (data[column] == 0).all()):
            dropped_features.append(data[[column]])
            data.drop([column], axis=1, inplace=True)

    data = (data - data.mean())/data.std()
    
    # restore dropped columns
    for column in dropped_features:
        data = data.merge(column, left_index=True, right_index=True)

    # restore column types
    data = data.astype(dtypes)

    return data

def features_chi_score_pd(data, target_feature, verbose):
    """
    Computes chi-squared stats between each non-negative feature and the class
    for the provided samples dataframe and plots the histogram of features scores.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])
    X.drop(X.columns[(X < 0).any()], axis=1, inplace=True)

    y = np.array(data[target_feature]).astype('bool')

    return features_chi_score_np(X, y, verbose)

def features_chi_score_np(X, y, verbose):
    """
    Computes chi-squared stats between each non-negative feature and the class
    for the provided samples numpy arrays and plots the histogram of features scores.
    """
    chi_score = SelectKBest(chi2, k=len(X.columns))
    chi_score.fit_transform(X, y)

    chiscore_scores = np.nan_to_num(chi_score.scores_)
    chiscore_indices = np.argsort(chiscore_scores)[::-1][0:len(X.columns)]
    chiscore_scores = chiscore_scores[chiscore_indices]
    norm_nom = (chiscore_scores - chiscore_scores.min())
    norm_denom = (chiscore_scores.max() - chiscore_scores.min())
    norm_chiscore_scores = norm_nom/norm_denom

    if verbose:
        plot_features_scores(X.columns, norm_chiscore_scores, 'Features Chi-Square Scores')
    
    return dict(zip(X.columns, norm_chiscore_scores))

def chi_square_statistic(data):
    """
    Computes the chi-square statistics between any two features in the given data.
    """
    chi_zeros=[(0 for i in range(len(data.columns))) for i in range(len(data.columns))]
    resultant_chi = pd.DataFrame(data=chi_zeros, columns=list(data.columns))
    resultant_chi.set_index(pd.Index(list(data.columns)), inplace = True)

    p_zeros=[(0 for i in range(len(data.columns))) for i in range(len(data.columns))]
    resultant_p = pd.DataFrame(data=p_zeros, columns=list(data.columns))
    resultant_p.set_index(pd.Index(list(data.columns)), inplace = True)

    for i in list(data.columns):
        for j in list(data.columns):
            if i != j:
                chi2_val, p_val = chi2(np.array(data[i]).reshape(-1, 1), np.array(data[j]).reshape(-1, 1))
                resultant_chi.loc[i,j] = chi2_val
                resultant_p.loc[i,j] = p_val

    fig = plt.figure(figsize=(16,16), dpi=100, facecolor='w', edgecolor='k')
    sns.heatmap(resultant_chi, annot=True, cmap='Blues')
    fig.suptitle('Chi-Square Test Results', fontsize=20, y=0.91)
    plt.show()

    fig = plt.figure(figsize=(16,16), dpi=100, facecolor='w', edgecolor='k')
    sns.heatmap(resultant_p, annot=True, cmap='Blues')
    fig.suptitle('Chi-Square Test p-value Results', fontsize=20, y=0.91)
    plt.show()

def features_mi_score_pd(data, target_feature, verbose):
    """
    Computes mutual information between each feature and the class for the
    provided samples dataframe and plots the histogram of features scores.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])

    y = np.array(data[target_feature]).astype('bool')

    return features_mi_score_np(X, y, verbose)

def features_mi_score_np(X, y, verbose):
    """
    Computes mutual information between each feature and the class for the
    provided samples numpy arrays and plots the histogram of features scores.
    """
    mi_score = SelectKBest(mutual_info_classif, k=len(X.columns))
    mi_score.fit_transform(X, y)

    miscore_scores = np.nan_to_num(mi_score.scores_)
    miscore_indices = np.argsort(miscore_scores)[::-1][0:len(X.columns)]
    miscore_scores = miscore_scores[miscore_indices]
    norm_nom = (miscore_scores - miscore_scores.min())
    norm_denom = (miscore_scores.max() - miscore_scores.min())
    norm_miscore_scores = norm_nom/norm_denom

    if verbose:
        plot_features_scores(X.columns, norm_miscore_scores, 'Features Mutual Information Scores')
    
    return dict(zip(X.columns, norm_miscore_scores))

def mean_impurity_decrease_pd(data, target_feature, estimators, verbose):
    """
    Computes the mean impurity decrease for each feature for the provided samples
    dataframe and plots the histogram of features scores.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])

    y = np.array(data[target_feature]).astype('bool')

    return mean_impurity_decrease_np(X, y, estimators, verbose)

def mean_impurity_decrease_np(X, y, estimators, verbose):
    """
    Computes the mean impurity decrease for each feature for the provided samples
    numpy arrays and plots the histogram of features scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    forest = RandomForestClassifier(n_estimators=estimators, n_jobs=-1)
    forest = forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    if verbose:
        plot_impurity_decrease(feature_importances, std)
    
    return dict(zip(X.columns, feature_importances))

def pca_features(data, target_feature):
    """
    Compute the 3 principal components for the given dataset.
    """
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, data[[target_feature]]], axis=1)
    return finalDf

def features_f_score_pd(data, target_feature, verbose):
    """
    Computes the ANOVA F-value for the provided samples dataframe
    and plots the histogram of features scores.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])

    y = np.array(data[target_feature]).astype('bool')

    return features_f_score_np(X, y, verbose)

def features_f_score_np(X, y, verbose):
    """
    Computes the ANOVA F-value for the provided samples numpy arrays
    and plots the histogram of features scores.
    """
    fscore = SelectKBest(f_classif, k=len(X.columns))
    fscore.fit_transform(X, y)

    fscore_scores = np.nan_to_num(fscore.scores_)
    fscore_indices = np.argsort(fscore_scores)[::-1][0:len(X.columns)]
    fscore_scores = fscore_scores[fscore_indices]
    norm_nom = (fscore_scores - fscore_scores.min())
    norm_denom = (fscore_scores.max() - fscore_scores.min())
    norm_fscore_scores = norm_nom/norm_denom

    if verbose:
        plot_features_scores(X.columns, norm_fscore_scores, 'Features F-Score Scores')
    
    return dict(zip(X.columns, norm_fscore_scores))

def features_woe_iv_scores_pd(data, target_feature, verbose):
    """
    Applies the Fayyad and Irani discretization and computes the WoE-IV score
    for the features of the given pandas dataframe.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])
    X = X[X.columns.drop(list(X.filter(regex='-woe')))]

    y = np.array(data[target_feature]).astype('bool')

    return features_woe_iv_scores_np(X, y, verbose)

def features_woe_iv_scores_np(X, y, verbose):
    """
    Applies the Fayyad and Irani discretization and computes the WoE-IV score
    for the features of the given numpy arrays.
    """
    iv_scores = {}
    discretized_data = fayyad_irani_discretization_np(X, y, verbose)

    for column in discretized_data.dtypes.index:
        woe_crosstab = ((pd.crosstab(discretized_data[column], y, normalize='columns')+laplacian_smoothing)
                            .assign(woe=lambda dfx: np.log(dfx[True] / dfx[False]))
                            .assign(iv=lambda dfx: np.sum(dfx['woe']*(dfx[True] - dfx[False]))))

        if verbose:
            plt.style.use("default")
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')
            woe_crosstab['woe'].plot(kind='hist', ax=ax)
            ax.set_xlabel("Values", fontsize=12, labelpad=10)
            ax.set_ylabel("Density", fontsize=12, labelpad=10)
            fig.suptitle(woe_crosstab.index.name, fontsize=15, y=0.95)
            plt.show()

        iv_scores[column] = woe_crosstab['iv'].values[0]

    if verbose:
        plot_features_scores(data.columns, list(iv_scores.values()), 'Features Information Value Scores')

    return iv_scores

def fayyad_irani_discretization_pd(data, target_feature, verbose):
    """
    Applies Fayyad and Irani discretization to the given input data (pandas dataframe).
    Fayyad and Irani discretization is an entropy based supervised and local
    discretization method.
    """
    X = data.drop([target_feature], axis=1, inplace=False)
    X = X.select_dtypes(include=['float64', 'int64', 'bool'])
    X.drop(X.columns[(X < 0).any()], axis=1, inplace=True)

    y = np.array(data[target_feature]).astype('bool')

    return fayyad_irani_discretization_np(X, y, verbose)

def fayyad_irani_discretization_np(X, y, verbose):
    """
    Applies Fayyad and Irani discretization to the given input data (numoy arrays).
    """
    print("Running Fayyad and Irani discretization.")

    discretizer = MDLP()
    discretizer.fit(X, y)
    discretized_data = discretizer.transform(X)

    if verbose:
        for i in range(X.shape[1]):
            count = Counter((round(elem[0], 4), round(elem[1], 4)) for elem in discretizer.cat2intervals(discretized_data, i))
            labels = [str(label) for label in count.keys()]
            values = [val for val in count.values()]
            plot_hist(labels, values, data.columns[i] + ' Discretization', 'Bins', 'Values Count', rotated_ticks=True)

    return pd.DataFrame(discretized_data, columns=X.columns)