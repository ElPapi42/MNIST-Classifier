import tensorflow as tf
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image

from DigitsClassifier import DigitsClassifier

#Load Train Data
(train_digits, train_labels), (eval_digits, eval_labels) = tf.keras.datasets.mnist.load_data("./Resources")
kaggle_digits = pd.read_csv("./Resources/test.csv").values

#Preprocess
train_digits = np.reshape(train_digits, [np.shape(train_digits)[0], 28, 28, 1])/255.0
eval_digits = np.reshape(eval_digits, [np.shape(eval_digits)[0], 28, 28, 1])/255.0
kaggle_digits = np.reshape(kaggle_digits, [np.shape(kaggle_digits)[0], 28, 28, 1])/255.0

#Generator
def get_sample(digits, return_labels=False, labels=None):
    if(return_labels):
        if(np.shape(digits)[0] == np.shape(labels)[0]):
            for index in range(0, np.shape(digits)[0]):
                yield (digits[index], labels[index])
        else:
            raise ValueError("Digits and Labels dont have the same numberof samples")
    else:
        for index in range(0, np.shape(digits)[0]):
            yield (digits[index])

def transform_sample(digit, label):
    rot = random.randint(-1, 2)
    t_digit = digit
    t_digit = tf.compat.v2.image.rot90(t_digit, rot)

    return t_digit, label

#Define datasets
train_ds = tf.data.Dataset.from_generator(get_sample, (tf.float32, tf.int32), args=[train_digits, True, train_labels]).map(transform_sample, 100).batch(1000).prefetch(2)
eval_ds = tf.data.Dataset.from_generator(get_sample, (tf.float32, tf.int32), args=[eval_digits, True, eval_labels]).batch(1000).prefetch(2)
kaggle_ds = tf.data.Dataset.from_generator(get_sample, (tf.float32), args=[kaggle_digits]).batch(1000).prefetch(2)

"""
for digits, label in train_ds.take(1):
    print(label)
    sns.regplot(data=digits)
    plt.show()
"""

#Define model and load weights (Pretrained on google colab notebook)
model = DigitsClassifier()
model.compile(tf.keras.optimizers.Adadelta(7.0), tf.keras.losses.SparseCategoricalCrossentropy(), [tf.keras.metrics.SparseCategoricalAccuracy()])
model.train_on_batch(train_ds.take(1))
model.load_weights("./Weights/06/weights")
print(model.summary())
#model.fit(train_ds, epochs=50, verbose=2, validation_data=eval_ds)

index = 1
results = [[], []]
for test_batch in kaggle_ds:
    results[0] = tf.concat([results[0], range(index, index + 1000)], axis=0)
    results[1] = tf.concat([results[1], tf.argmax(model(test_batch), axis=1)], axis=0)
    index += 1000

results_df = pd.DataFrame({"ImageId":results[0], "Label":results[1]}, dtype=np.int32)
results_df.to_csv("./Results/06/results.csv", index=False) 
