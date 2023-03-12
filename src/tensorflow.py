#!/usr/bin/env python

"""
tensorflow.py: Implementation of utility functions for Tensorflow models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, losses, callbacks, regularizers, Input, Model

class LearningRateDecay:
    def plot(self, epochs, figsize=(7, 7), save_path=None, dpi=100, title="Learning Rate Decay"):
        # compute the set of learning rates for each corresponding epoch
        lrs = [self(i) for i in epochs]

        # plot
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")

        if save_path is not None:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")

        plt.show()

class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.001, factor=0.6, dropEvery=20):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)

class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay

        # return the new learning rate
        return float(alpha)

def sequential_model(name):
    """
    Returns an initialized tf.keras.models.Sequential model.
    """
    return models.Sequential(name=name)

def input_layer(input_shape, dtype):
    """
    Returns a tf.keras.layers.InputLayer initialized with the given parameters.
    """
    return layers.InputLayer(input_shape=input_shape, dtype=dtype)

def dense_layer(units, activation, name=None, kernel_regularizer=None, use_bias=True):
    """
    Returns a tf.keras.layers.Dense layer with default kernel and bias initializer.
    """
    return layers.Dense(units=units, activation=activation,
                        kernel_initializer='random_normal',
                        bias_initializer='zeros',
                        kernel_regularizer=kernel_regularizer,
                        use_bias=use_bias,
                        name=name)

def learningRateSchedulerCallback(schedule):
    """
    """
    return callbacks.LearningRateScheduler(schedule=schedule)

def modelCheckpointCallback(filepath, monitor, save_best_only, mode, verbose):
    """
    """
    return callbacks.ModelCheckpoint(filepath=filepath, monitor=monitor,
                                     save_best_only=save_best_only, mode=mode,
                                     verbose=verbose)

def earlyStoppingCallback(monitor, min_delta, mode, patience, verbose):
    """
    """
    return callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                   patience=patience, verbose=verbose, mode=mode)

def RMSpropOptimizer(learning_rate):
    """
    """
    return optimizers.RMSprop(learning_rate=learning_rate)

def AdamOptimizer(learning_rate):
    """
    """
    return optimizers.Adam(learning_rate=learning_rate)

def AdadeltaOptimizer(learning_rate):
    """
    """
    return optimizers.Adadelta(learning_rate=learning_rate)

def BinaryCrossentropyLoss():
    """
    """
    return losses.BinaryCrossentropy()

def BinaryFocalCrossentropy(apply_class_balancing, alpha):
    """
    """
    return losses.BinaryFocalCrossentropy(apply_class_balancing=apply_class_balancing,
                                                   alpha=alpha)

def plot_model(model, to_file, show_layer_activations, show_shapes, rankdir, dpi):
    """
    """
    tf.keras.utils.plot_model(model=model, to_file=to_file,
                              show_layer_activations=show_layer_activations,
                              show_shapes=show_shapes, rankdir=rankdir, dpi=dpi)

def one_hot(indices, depth):
    """
    """
    return tf.one_hot(indices, depth)

def clear_session():
    """
    """
    tf.keras.backend.clear_session()

"""
PointNet
"""
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(regularizers.Regularizer):
    def __init__(self, num_features, l2=0.001):
        self.num_features = num_features
        self.eye = tf.eye(num_features)
        self.l2 = l2

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2 * tf.square(xxt - self.eye))
    
    def get_config(self):
        return {'l2': float(self.l2), 'num_features': int(self.num_features)}

def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)

    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def point_net_model():
    # define model to be trained and tested
    inputs = Input(shape=(11, 1))

    # layers
    x = tnet(inputs, 1)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    # output softmax classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # define model
    return Model(inputs=inputs, outputs=outputs, name="PointNet")