#!/usr/bin/env python

"""
tensorflow.py: Implementation of utility functions for Tensorflow models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import copy
import logging
import itertools
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy import interp
from itertools import cycle
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from IPython.display import Markdown, display
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

def input_layer(input_shape, dtype):
    """
    Returns a tf.keras.layers.InputLayer initialized with the given parameters.
    """
    return tf.keras.layers.InputLayer(input_shape=input_shape, dtype=dtype)

def dense_layer(units, activation, name=None, kernel_regularizer=None, use_bias=True):
    """
    Returns a tf.keras.layers.Dense layer with default kernel and bias initializer.
    """
    return tf.keras.layers.Dense(units=units, activation=activation,
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=kernel_regularizer,
                                 use_bias=use_bias,
                                 name=name)

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