import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mnist = pd.read_csv('./Resources/train.csv')

labels = mnist[["label"]].values
labels = np.squeeze(labels)

digits = mnist.drop(["label"], axis=1).values
digits = np.reshape(digits, [np.shape(digits)[0], 28, 28])
digits = digits/255

dataset = tf.data.Dataset.from_tensor_slices((digits, labels)).batch(1)