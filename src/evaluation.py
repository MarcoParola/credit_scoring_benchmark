#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for evaluation models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import gc

import os

import numpy as np

import pandas as pd

from tqdm import tqdm

from src import pytorch
from src import plotting
from src import tensorflow

import tensorflow as tf
from tensorflow.keras import models, regularizers, optimizers

import numpy as np
from numpy import vstack

from torch.optim import SGD
from torch.nn import BCELoss

from datetime import datetime

from collections import Counter

from IPython.display import Markdown, display

from EMP.metrics import empCreditScoring

from sklearn import metrics
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
    if verbose:
        plotting.plot_features_scores(list(features_scores.keys())[:k],
                             list(features_scores.values())[:k],
                             '',
                             save_path=save_path, dpi=200)

    return list(features_scores.keys())[:k]

def gini_score(y_true, y_prob):
    """
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
    """
    return gini_score(y_true, y_prob) / gini_score(y_true, y_true)

def brier_score(y_true, y_prob):
    """
    """
    return metrics.brier_score_loss(y_true, y_prob)

def custom_brier_score(y_true, y_prob):
    """
    """
    assert (len(y_true) == len(y_prob))
    losses = np.subtract(y_true, y_prob)**2
    brier_score = losses.sum()/len(y_true)
    return brier_score

def emp_score(y_true, y_prob):
    """
    """
    assert (len(y_true) == len(y_prob))
    return empCreditScoring(y_prob, y_true, p_0=0.800503355704698, p_1=0.199496644295302, ROI=0.2644, print_output=False)[0]

def emp_score_frac(y_true, y_prob):
    """
    """
    assert (len(y_true) == len(y_prob))
    return empCreditScoring(y_prob, y_true, p_0=0.800503355704698, p_1=0.199496644295302, ROI=0.2644, print_output=False)

def confusion_matrix_report(model_name, conf_matrix_list, classes, save_path=None, dpi=100):
    """
    Plots average normalized confusion matrix.
    """
    mean_of_conf_matrix_list = np.mean(conf_matrix_list, axis=0)

    plotting.plot_confusion_matrix(cnf_matrix=mean_of_conf_matrix_list,
                                   classes=classes, normalize=True,
                                   title='Normalized confusion matrix',
                                   save_path=save_path, dpi=dpi)

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
    
def k_fold_cross_validate(layers, train_data, test_data, target, k_folds,
                          features_scores, features, model_name, learning_rate,
                          epochs, batch_size, classes, verbose):
    """
    Run k-fold cross validation on the Sequential model built using the given layers.
    """
    save_path = create_model_save_path(model_name)
    print('Model save_path: ' + save_path)

    # separate class label from other features
    train_y = np.array(train_data[target])
    train_X = train_data.drop([target], axis=1, inplace=False)
    test_y = np.array(test_data[target])
    if len(classes) > 2:
        test_y = tf.one_hot(test_y, len(classes))
    test_X = test_data.drop([target], axis=1, inplace=False)

    # store train and test metrics
    model_accuracies_train = []
    model_accuracies_val = []
    model_f1_train = []
    model_f1_val = []
    model_precision_train = []
    model_precision_val = []
    model_recall_train = []
    model_recall_val = []
    model_fpr = []
    model_tpr = []
    model_thresh = []
    model_auc = []
    model_conf_matrix_list = []
    gini_scores = []
    brier_scores = []
    emp_scores = []
    emp_fractions = []

    # current fold index
    fold_counter = 1

    # features selection
    if features > 0:
        k_best_features = select_k_best_features(features_scores, features, verbose)
        print('Selected Features: ' + str(k_best_features))
        test_X = test_X[k_best_features]
    test_X = test_X.to_numpy().reshape((test_X.shape[0], test_X.shape[1], 1))

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for train_index, validation_index in tqdm(skf.split(train_X, train_y)):
        if verbose:
            display(Markdown("# **FOLD " + str(fold_counter) + "**"))

        # training and validation data folds
        train_fold = train_data.iloc[train_index]
        train_fold_y = np.array(train_fold[target])
        if len(classes) > 2:
            train_fold_y = tf.one_hot(train_fold_y, len(classes))
        train_fold_X = train_fold.drop([target], axis=1, inplace=False)
        if features > 0:
            train_fold_X = train_fold_X[k_best_features]
        train_fold_X = train_fold_X.to_numpy().reshape((train_fold_X.shape[0], train_fold_X.shape[1], 1))

        validation_fold = train_data.iloc[validation_index]
        val_fold_y = np.array(validation_fold[target])
        if len(classes) > 2:
            val_fold_y = tf.one_hot(val_fold_y, len(classes))
        val_fold_X = validation_fold.drop([target], axis=1, inplace=False)
        if features > 0:
            val_fold_X = val_fold_X[k_best_features]
        val_fold_X = val_fold_X.to_numpy().reshape((val_fold_X.shape[0], val_fold_X.shape[1], 1))

        pred_train, pred_val, pred_test, pred_probs = train_tf_model(model_name, layers, classes, learning_rate, epochs,
                                                                     batch_size, save_path, fold_counter, train_fold_X,
                                                                     train_fold_y, val_fold_X, val_fold_y, test_X, test_y)

        # collect accuracy, f1, precision and recall statistics
        if len(classes) > 2:
            model_accuracies_train.append(metrics.accuracy_score(train_fold_y, pred_train))
            model_accuracies_val.append(metrics.accuracy_score(val_fold_y, pred_val))
            model_f1_train.append(metrics.f1_score(train_fold_y, pred_train, average='micro'))
            model_f1_val.append(metrics.f1_score(val_fold_y, pred_val, average='micro'))
            model_precision_train.append(metrics.precision_score(train_fold_y, pred_train, average='micro'))
            model_precision_val.append(metrics.precision_score(val_fold_y, pred_val, average='micro'))
            model_recall_train.append(metrics.recall_score(train_fold_y, pred_train, average='micro'))
            model_recall_val.append(metrics.recall_score(val_fold_y, pred_val, average='micro'))
        else:
            model_accuracies_train.append(metrics.accuracy_score(train_fold_y, pred_train))
            model_accuracies_val.append(metrics.accuracy_score(val_fold_y, pred_val))
            model_f1_train.append(metrics.f1_score(train_fold_y, pred_train, pos_label=1))
            model_f1_val.append(metrics.f1_score(val_fold_y, pred_val, pos_label=1))
            model_precision_train.append(metrics.precision_score(train_fold_y, pred_train, pos_label=1))
            model_precision_val.append(metrics.precision_score(val_fold_y, pred_val, pos_label=1))
            model_recall_train.append(metrics.recall_score(train_fold_y, pred_train, pos_label=1))
            model_recall_val.append(metrics.recall_score(val_fold_y, pred_val, pos_label=1))

        # confusion matrix score
        if len(classes) > 2:
            model_conf_matrix = metrics.confusion_matrix(np.array(test_split[target]), np.argmax(model.predict(test_X), axis=-1))
            model_conf_matrix_list.append(model_conf_matrix)
        else:
            model_conf_matrix = metrics.confusion_matrix(test_y, pred_test)
            model_conf_matrix_list.append(model_conf_matrix)

        # collect roc curve statistics
        model_fpr_new, model_tpr_new, model_thresh_new = metrics.roc_curve(test_y, pred_probs, pos_label=1)
        model_fpr.append(model_fpr_new)
        model_tpr.append(model_tpr_new)
        model_thresh.append(model_thresh_new)
        model_auc_new = metrics.roc_auc_score(test_y, pred_test)
        model_auc.append(model_auc_new)

        # compute fold gini coefficient
        gini_scores.append(normalized_gini_score(test_y, pred_probs))

        # compute fold brier score
        brier_scores.append(brier_score(test_y, pred_probs))

        # compute fold missclass error costs
        emp = emp_score_frac(test_y, pred_probs)
        emp_scores.append(emp.EMPC)
        emp_fractions.append(emp.EMPC_fraction)

        if verbose:
            # current model report
            print(metrics.classification_report(train_fold_y, pred_train, labels=[0,1]))
            print(metrics.classification_report(val_fold_y, pred_val, labels=[0,1]))

        print("\n-------- TERMINATED FOLD: " + str(fold_counter) + " --------")

        fold_counter += 1

    # store metrics as csv file
    metrics_dict = {'accuracy': model_accuracies_val,
                    'f1': model_f1_val,
                    'precision': model_precision_val,
                    'recall': model_recall_val,
                    'auc': model_auc,
                    'gini': gini_scores,
                    'brier': brier_scores,
                    'emp': emp_scores,
                    'emp_frac': emp_fractions}
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(save_path + '/metrics.csv', index=False)

    # accuracy report and plotaccuracy_report
    plotting.plot_accuracy(model_name = '',
                           accuracies_train = model_accuracies_train,
                           accuracies_val = model_accuracies_val,
                           save_path = save_path + '/' + model_name + '-',
                           dpi = 100)

    # f1 score and plot
    plotting.plot_f1(model_name = '',
                     f1_train = model_f1_train,
                     f1_val = model_f1_val,
                     save_path = save_path + '/' + model_name + '-',
                     dpi = 100)
    
    # precision score and plot
    plotting.plot_precision(model_name = '',
                    precision_train = model_precision_train,
                    precision_val = model_precision_val,
                    save_path = save_path + '/' + model_name + '-',
                    dpi = 100)

    # recall score and plot
    plotting.plot_recall(model_name = '',
                         recall_train = model_recall_train,
                         recall_val = model_recall_val,
                         save_path = save_path + '/' + model_name + '-',
                         dpi = 100)

    # confusion matrix
    confusion_matrix_report(model_name = '',
                            conf_matrix_list = model_conf_matrix_list,
                            classes = classes,
                            save_path = save_path + '/' + model_name + '-',
                            dpi = 100)

    # roc curves and auc scores
    plotting.plot_roc_auc_scores(model_name = '',
                                 fprs = model_fpr,
                                 tprs = model_tpr,
                                 thresholds = model_thresh,
                                 save_path = save_path + '/' + model_name + '-',
                                 dpi = 100)
    
    # plot folds gini scores
    plotting.plot_gini(model_name = '',
                       gini_scores = gini_scores,
                       save_path = save_path + '/' + model_name + '-',
                       dpi = 100)

    # plot folds brier scores
    plotting.plot_brier(model_name = '',
                        brier_scores = brier_scores,
                        save_path = save_path + '/' + model_name + '-',
                        dpi = 100)

    # plot folds error costs
    plotting.plot_emp(model_name = '',
                      emp_scores = emp_scores,
                      emp_fractions = emp_fractions,
                      save_path = save_path + '/' + model_name + '-',
                      dpi = 100)

def train_tf_model(model_name, layers, classes, learning_rate, epochs, batch_size, save_path,
                   fold_counter, train_fold_X, train_fold_y, val_fold_X, val_fold_y, test_X, test_y):
    """
    """
    # best fold model save path
    fold_model_save_path = os.path.join(save_path, 'fold-' + str(fold_counter))

    # learning rate scheduler, best model and early stopping callbacks
    step_decay_schedule = tensorflow.StepDecay(initAlpha=learning_rate, factor=0.9, dropEvery=30)
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule)
    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fold_model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    val_loss_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=50, verbose=1)

    # add callbacks
    callbacks_list = [best_model_checkpoint, lr_scheduler_callback, val_loss_early_stopping]

    # define model to be trained and tested
    model = models.Sequential(name=model_name + "-" + str(fold_counter))
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
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    #loss_function = tf.keras.losses.BinaryCrossentropy()
    #loss_function = tf.keras.losses.BinaryFocalCrossentropy()
    #loss_function = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)
    loss_function = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.70)
    #loss_function = tfa.losses.ContrastiveLoss()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=["accuracy"])
    tf.keras.utils.plot_model(model, to_file=save_path+'/network.pdf',
                              show_layer_activations=True, show_shapes=True,
                              rankdir="TB", dpi=200)

    # train model
    history = model.fit(train_fold_X,
                        train_fold_y,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        batch_size=batch_size,
                        validation_data=(val_fold_X, val_fold_y))

    # plot learning rate decay
    step_decay_schedule.plot(np.arange(0, epochs), figsize=(7, 7),
                             save_path=save_path+'/learning-rate-decay.pdf', dpi=100)

    # plot loss and acuracy for each training/validation fold
    plotting.plot_train_loss_accuracy(history.history["loss"], history.history["accuracy"],
                             history.history["val_loss"], history.history["val_accuracy"],
                             save_path = save_path + '/' + model_name + '-' + str(fold_counter) + '-',
                             dpi = 100)

    # load best fold model
    model.load_weights(fold_model_save_path)

    # evaluate the best fold model on the test set
    test_loss, test_accuracy = model.evaluate(test_X, test_y, batch_size=batch_size)

    # test model on train and test set
    if len(classes) > 2:
        pred_train = np.argmax(model.predict(train_fold_X), axis=-1)
        pred_train = tf.one_hot(pred_train, len(classes))
        pred_val = np.argmax(model.predict(val_fold_X), axis=-1)
        pred_val = tf.one_hot(pred_val, len(classes))
        pred_test = np.argmax(model.predict(test_X), axis=-1)
        pred_test = tf.one_hot(pred_test, len(classes))
        pred_probs = np.argmax(model.predict(test_X), axis=-1)
    else:
        pred_train = (model.predict(train_fold_X) > 0.5).astype("int32")
        pred_val = (model.predict(val_fold_X) > 0.5).astype("int32")
        pred_test = (model.predict(test_X) > 0.5).astype("int32")
        pred_probs = np.concatenate(model.predict(test_X))

    # clean up model
    model = None
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return pred_train, pred_val, pred_test, pred_probs

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