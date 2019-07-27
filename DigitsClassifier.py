import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DigitsClassifier(tf.keras.Model):
    """MNIST Digit Classifier"""

    def __init__(self):
        super(DigitsClassifier, self).__init__(self)

        self.conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", input_shape=[28, 28, 1])
        self.pooling_0 = tf.keras.layers.MaxPool2D(2)
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.pooling_1 = tf.keras.layers.MaxPool2D(4)

        self.flatten = tf.keras.layers.Flatten()

        self.hiden = tf.keras.layers.Dense(10, "relu")
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):

        output = self.conv_0(inputs)
        output = self.pooling_0(output)
        output = self.conv_1(output)
        output = self.pooling_1(output)

        output = self.flatten(output)
        output = self.hiden(output)
        output = self.softmax(output)

        return output




