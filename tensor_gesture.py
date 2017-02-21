# This file is used for training the model for
# recognizing gestures.
#
# The classifier I choose is the DNNClassifier
# in Tensorflow.
#
# In the training procedure, I use 5000 steps to
# learn the traning datasets.
# ===============================================


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

# Load data sets.
COLUMNS = ["-10", "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2",
            "3", "4", "5", "6", "7", "8", "9", "10", "number"]
FEATURES = ["-10", "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2",
            "3", "4", "5", "6", "7", "8", "9", "10"]
LABEL = "number"

training_set = pd.read_csv("gesture_data_train.csv", skipinitialspace=True,
                            skiprows=1, names=COLUMNS)
test_set = pd.read_csv("gesture_data_test.csv", skipinitialspace=True,
                        skiprows=1, names=COLUMNS)

# Defining feature and create the DNNClassifier.
# Specify that all features have real-value datasets.
feature_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                            hidden_units=[10, 20, 10],
                                            n_classes=5,
                                            model_dir="gesture_model")
# Build the input_fn.
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1])
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values, shape=[data_set[k].size, 1])
    return feature_cols, labels

# Training the classifier.
classifier.fit(input_fn=lambda: input_fn(training_set), steps=2000)

# Evaluate the mod.
ev = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
# Retrieve the loss from the ev results and print it to output.
loss_score = ev['loss']
print("Loss: {0:f}".format(loss_score))
