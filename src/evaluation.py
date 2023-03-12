#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for evaluation models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import gc
import os
import numpy as np
from tqdm import tqdm

from src import pytorch
from src import plotting
from src import tensorflow
from src import metrics

import numpy as np
from numpy import vstack

from torch.optim import SGD
from torch.nn import BCELoss

from datetime import datetime
from IPython.display import Markdown, display
from sklearn.model_selection import StratifiedKFold

def is_dl_model(model):
    """
    Checks if the given model is based on PyTorch.
    """
    return 'pytorch' in model.__class__.__module__

def select_k_best_features(features_scores, k, verbose=True, save_path=None):
    """
    Selects the k best features based on the scores obtained using different
    features slection techniques.
    """
    features_scores = sorted(features_scores.items(), key=lambda x: x[1], reverse=True)
    features_scores = dict(features_scores)

    if verbose:
        plotting.plot_features_scores(list(features_scores.keys())[:k],
                             list(features_scores.values())[:k],
                             '',
                             save_path=save_path, dpi=200)

    return list(features_scores.keys())[:k]

def create_model_save_path(model_name):
    """
    Creates train and test results save directory for the given
    model name.
    """
    ret = '../../../models'
    if not os.path.isdir(ret):
        os.mkdir(ret)

    ret = os.path.join(ret, model_name)
    if not os.path.isdir(ret):
        os.mkdir(ret)

    ret = os.path.join(ret, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.isdir(ret):
        os.mkdir(ret)

    return ret

def collect_perf_metrics(metrics_dict, y_train, pred_train, y_val, pred_val,
                         y_test, pred_test, test_probs, classes):
    """
    Computes and collects performance metrics based on the given predictions and
    groudtruth.
    """
    if not len(metrics_dict) > 0:
        metrics_dict = {'train_accuracy':[], 'train_f1':[], 'train_precision':[], 'train_recall':[],
                        'val_accuracy':[], 'val_f1':[], 'val_precision':[], 'val_recall':[],
                        'test_accuracy':[], 'test_f1':[], 'test_precision':[], 'test_recall':[],
                        'fpr':[], 'tpr':[], 'thresh':[], 'auc':[], 'conf_matrix':[], 'gini':[],
                        'brier':[], 'h_measure':[], 'ks_statistic':[], 'ks_pvalue':[],
                        'emp_score':[], 'emp_frac':[]}

    metrics_dict['train_accuracy'].append(metrics.accuracy_score(y_train, pred_train, classes))
    metrics_dict['train_f1'].append(metrics.f1_score(y_train, pred_train, classes))
    metrics_dict['train_precision'].append(metrics.precision_score(y_train, pred_train, classes))
    metrics_dict['train_recall'].append(metrics.recall_score(y_train, pred_train, classes))

    metrics_dict['val_accuracy'].append(metrics.accuracy_score(y_val, pred_val, classes))
    metrics_dict['val_f1'].append(metrics.f1_score(y_val, pred_val, classes))
    metrics_dict['val_precision'].append(metrics.precision_score(y_val, pred_val, classes))
    metrics_dict['val_recall'].append(metrics.recall_score(y_val, pred_val, classes))

    metrics_dict['test_accuracy'].append(metrics.accuracy_score(y_test, pred_test, classes))
    metrics_dict['test_f1'].append(metrics.f1_score(y_test, pred_test, classes))
    metrics_dict['test_precision'].append(metrics.precision_score(y_test, pred_test, classes))
    metrics_dict['test_recall'].append(metrics.recall_score(y_test, pred_test, classes))

    metrics_dict['conf_matrix'].append(metrics.cm_score(y_test, pred_test, classes))        

    model_fpr_new, model_tpr_new, model_thresh_new = metrics.roc_curve(y_test, test_probs)
    metrics_dict['fpr'].append(model_fpr_new)
    metrics_dict['tpr'].append(model_tpr_new)
    metrics_dict['thresh'].append(model_thresh_new)
    model_auc_new = metrics.roc_auc_score(y_test, pred_test)
    metrics_dict['auc'].append(model_auc_new)

    metrics_dict['gini'].append(metrics.normalized_gini_score(y_test, pred_test))
    metrics_dict['brier'].append(metrics.brier_score(y_test, pred_test))
    metrics_dict['h_measure'].append(metrics.h_measure(y_test, pred_test))
    ks = metrics.ks_score(y_test, test_probs)
    metrics_dict['ks_statistic'].append(ks[0])
    metrics_dict['ks_pvalue'].append(ks[1])
    emp = metrics.emp_score_frac(y_test, pred_test)
    metrics_dict['emp_score'].append(emp.EMPC)
    metrics_dict['emp_frac'].append(emp.EMPC_fraction)

    return metrics_dict

def k_fold_cross_validate(clf, layers, train_data, test_data, target, classes, k_folds,
                          features_scores, features, model_name, learning_rate,
                          epochs, batch_size, verbose):
    """
    Runs k-fold cross validation on the model.
    """
    save_path = create_model_save_path(model_name)
    print('Model save_path: ' + save_path)

    if layers is not None:
        k_fold_cross_validate_dl_model(layers, train_data, test_data, target, classes,
                                       k_folds, features_scores, features, model_name,
                                       learning_rate, epochs, batch_size, save_path,
                                       verbose)
    else:
        k_fold_cross_validate_ml_model(clf, train_data, test_data, target, classes,
                                       k_folds, features_scores, features, model_name,
                                       save_path, verbose)

def k_fold_cross_validate_ml_model(clf, train_data, test_data, target, classes,
                                   k_folds, features_scores, features, model_name,
                                   save_path, verbose):
    """
    Performs K-fold Cross Validation using the given model on the given dataset.
    """
    metrics_dict = {}

    # separate class label from other features
    train_labels = np.array(train_data[target])
    train_data = train_data.drop([target], axis=1, inplace=False)

    y_test = np.array(test_data[target])
    X_test = test_data.drop([target], axis=1, inplace=False)

    # current fold index
    fold_counter = 1

    # features selection
    if features > 0:
        k_best_features = select_k_best_features(features_scores, features, verbose,
                                                 save_path = save_path + '/' + model_name + '-',)
        print('Selected Features: ' + str(k_best_features))

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for train_index, val_index in tqdm(skf.split(train_data, train_labels)):
        if verbose:
            display(Markdown("# **FOLD " + str(fold_counter) + "**"))

        # split data in train and test set
        X_train, X_val = train_data.iloc[train_index], train_data.iloc[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]

        # only use high scoring features
        if features > 0:
            X_train = X_train[k_best_features]
            X_val = X_val[k_best_features]
            X_test = X_test[k_best_features]

        # train model
        clf = clf.fit(X_train, y_train)

        # test model on train and test set
        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)
        pred_test = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)[:, 1]

        metrics_dict = collect_perf_metrics(metrics_dict, y_train, pred_train, y_val,
                                            pred_val, y_test, pred_test, test_probs, classes)

        if verbose:
            metrics.classification_report(y_train, pred_train, y_val, pred_val)

        fold_counter += 1

    metrics.report_performance_metrics(metrics_dict, save_path, model_name, classes)

def k_fold_cross_validate_dl_model(layers, train_data, test_data, target, classes, k_folds,
                                   features_scores, features, model_name, learning_rate,
                                   epochs, batch_size, save_path, verbose):
    """
    Runs k-fold cross validation on the Sequential model built using the given layers.
    """
    metrics_dict = {}

    # separate class label from other features
    train_labels = np.array(train_data[target])
    train_data = train_data.drop([target], axis=1, inplace=False)

    y_test = np.array(test_data[target])
    if len(classes) > 2:
        y_test = tensorflow.one_hot(y_test, len(classes))
    X_test = test_data.drop([target], axis=1, inplace=False)

    # current fold index
    fold_counter = 1

    # features selection
    if features > 0:
        k_best_features = select_k_best_features(features_scores, features, verbose)
        print('Selected Features: ' + str(k_best_features))
        X_test = X_test[k_best_features]
    X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for train_index, val_index in tqdm(skf.split(train_data, train_labels)):
        if verbose:
            display(Markdown("# **FOLD " + str(fold_counter) + "**"))

        # training and validation data folds
        y_train = train_labels[train_index]
        if len(classes) > 2:
            y_train = tensorflow.one_hot(y_train, len(classes))
        X_train = train_data.iloc[train_index]
        if features > 0:
            X_train = X_train[k_best_features]
        X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))

        y_val = train_labels[val_index]
        if len(classes) > 2:
            y_val = tensorflow.one_hot(y_val, len(classes))
        X_val = train_data.iloc[val_index]
        if features > 0:
            X_val = X_val[k_best_features]
        X_val = X_val.to_numpy().reshape((X_val.shape[0], X_val.shape[1], 1))

        pred_train, pred_val, pred_test, test_probs = train_tf_model(model_name, layers, classes, learning_rate, epochs,
                                                                     batch_size, save_path, fold_counter, X_train,
                                                                     y_train, X_val, y_val, X_test, y_test)

        metrics_dict = collect_perf_metrics(metrics_dict, y_train, pred_train, y_val,
                                            pred_val, y_test, pred_test, test_probs, classes)

        if verbose:
            metrics.classification_report(y_train, pred_train, y_val, pred_val)

        print("\n-------- TERMINATED FOLD: " + str(fold_counter) + " --------")

        fold_counter += 1

    metrics.report_performance_metrics(metrics_dict, save_path, model_name, classes)

def train_tf_model(model_name, layers, classes, learning_rate, epochs, batch_size, save_path,
                   fold_counter, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains a Tensorflow model and returns predictions on train, validation and test
    sets.
    """
    # best fold model save path
    fold_model_save_path = os.path.join(save_path, 'fold-' + str(fold_counter))

    # learning rate scheduler, best model and early stopping callbacks
    lr_scheduler = tensorflow.StepDecay(initAlpha=learning_rate, factor=0.9, dropEvery=30)
    lrs_callback = tensorflow.learningRateSchedulerCallback(lr_scheduler)
    checkpoint_callback = tensorflow.modelCheckpointCallback(fold_model_save_path, monitor='val_loss',
                                                             save_best_only=True, mode='min', verbose=1)
    early_stopping_callback = tensorflow.earlyStoppingCallback(monitor='val_loss', mode='min',
                                                               min_delta=0.001, patience=50, verbose=1)

    # add callbacks
    callbacks_list = [checkpoint_callback, lrs_callback, early_stopping_callback]

    # define model to be trained and tested
    model = tensorflow.sequential_model(name=model_name + "-" + str(fold_counter))
    for layer in layers:
        model.add(layer)

    # make sure models weights are initialed randomly at each fold
    if fold_counter == 1:
        model.save_weights(save_path + '/weights.h5')
    else:
        model.load_weights(save_path + '/weights.h5')

    # print model summary
    model.summary()

    # compile model
    optimizer = tensorflow.RMSpropOptimizer(learning_rate=learning_rate)
    loss_function = tensorflow.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.70)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=["accuracy"])
    tensorflow.plot_model(model, to_file=save_path+'/network.pdf', show_layer_activations=True,
                          show_shapes=True, rankdir="TB", dpi=200)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val))

    # plot learning rate decay
    lr_scheduler.plot(np.arange(0, epochs), figsize=(7, 7),
                      save_path=save_path+'/learning-rate-decay.pdf', dpi=100)

    # plot loss and acuracy for each training/validation fold
    plotting.plot_train_loss_accuracy(history.history["loss"], history.history["accuracy"],
                             history.history["val_loss"], history.history["val_accuracy"],
                             save_path = save_path + '/' + model_name + '-' + str(fold_counter) + '-',
                             dpi = 100)

    # load best fold model
    model.load_weights(fold_model_save_path)

    # evaluate the best fold model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test Loss: ' + str(test_loss))
    print('Test Accuracy: ' + str(test_accuracy))

    # test model on train, val and test sets
    if len(classes) > 2:
        pred_train = np.argmax(model.predict(X_train), axis=-1)
        pred_train = tensorflow.one_hot(pred_train, len(classes))
        pred_val = np.argmax(model.predict(X_val), axis=-1)
        pred_val = tensorflow.one_hot(pred_val, len(classes))
        pred_test = np.argmax(model.predict(X_test), axis=-1)
        pred_test = tensorflow.one_hot(pred_test, len(classes))
        test_probs = np.argmax(model.predict(X_test), axis=-1)
    else:
        pred_train = (model.predict(X_train) > 0.5).astype("int32")
        pred_val = (model.predict(X_val) > 0.5).astype("int32")
        pred_test = (model.predict(X_test) > 0.5).astype("int32")
        test_probs = np.concatenate(model.predict(X_test))

    # clean up model
    model = None
    del model
    tensorflow.clear_session()
    gc.collect()

    return pred_train, pred_val, pred_test, test_probs

def train_pt_model(train_dl, model):
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
def evaluate_pt_model(test_dl, model):
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