import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConvBlock(tf.keras.layers.Layer):
    """Convolutional Block featuring Conv2D + Pooling"""

    def __init__(self, conv_deep=1, kernels=32, kernel_size=3, pool_size=2, dropout_rate=1.0):
        super(ConvBlock, self).__init__(self)

        self.conv_layers = []
        self.pooling_layers = []
        self.bnorm_layers = []
        self.dropout_layers = []
        for index in range(0, conv_deep):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=kernels, kernel_size=kernel_size, padding="same", activation="relu"))
            self.pooling_layers.append(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            self.bnorm_layers.append(tf.keras.layers.BatchNormalization())
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(self, inputs, training=False):
        output = inputs

        for (conv, pooling, bnorm, dropout) in zip(self.conv_layers, self.pooling_layers, self.bnorm_layers, self.dropout_layers):
            output = conv(output)
            output = pooling(output)
            output = bnorm(output)

            if training:
                output = dropout(output)

        return output

class DigitsClassifier(tf.keras.Model):
    """MNIST Digit Classifier"""

    def __init__(self):
        super(DigitsClassifier, self).__init__(self)

        self.conv = ConvBlock(conv_deep=1, kernels=2)

        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Dense(10, "softmax")

    def call(self, inputs):

        output = self.conv(inputs)
        output = self.flatten(output)
        output = self.softmax(output)

        return output




