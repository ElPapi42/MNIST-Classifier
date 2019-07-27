import tensorflow as tf
import tensorflow.keras.datasets.mnist as mn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from DigitsClassifier import DigitsClassifier

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data("./Resources")
mnist_digits = np.concatenate([mnist_train[0], mnist_test[0]], axis=0)
mnist_labels = np.concatenate([mnist_train[1], mnist_test[1]], axis=0)

labels = mnist_labels

digits = mnist_digits
digits = np.reshape(digits, [np.shape(digits)[0], 28, 28, 1])
digits = digits/255

print(np.shape(digits))

def get_sample():
    for index in range(0, np.shape(labels)[0]):
        yield (digits[index], labels[index])


train_ds = tf.data.Dataset.from_generator(get_sample, (tf.float32, tf.int32))
test_ds = train_ds.take(10000).batch(1000).prefetch(2)
train_ds = train_ds.skip(10000).batch(1000).prefetch(2)

model = DigitsClassifier()
model.compile(tf.keras.optimizers.Adam(0.001), tf.keras.losses.SparseCategoricalCrossentropy())
model.fit(train_ds, epochs=50, verbose=2, validation_data=test_ds)

for digit, label in test_ds.take(1):
    print(tf.argmax(model(digit), axis=1), label)