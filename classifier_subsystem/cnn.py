"""
source: https://github.com/KyotoSunshine/CNN-for-handwritten-kanji/blob/master/model.py#L88
"""

# from typing import Any, Union
#
# import sklearn
# from pandas import Series, Index
# from pandas import DataFrame
# from pandas.core.arrays import ExtensionArray
# from pandas.core.generic import NDFrame
# from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
# from sklearn import tree
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
import os
from datetime import timedelta
import pickle as pkl
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import cv2
import os
from utils.params import *
import sys


import utils.classify_util as classify_utils

NUMBER_OF_CLASSES = 3166
BATCH_SIZE = 100
EPOCHS = 20
INITIAL_ADAM_LEARNING_RATE = 0.01
# If you don't mind long training times, make the below two values larger
MAXIMUM_NUMBER_OF_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5


class Flush(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        sys.stdout.flush()

        sys.stderr.flush()


def train_model( X_train, X_val, y_train, y_val):

    # load data
    X_train, X_val, y_train, y_val = classify_utils.load_train_data(clean=False)
    # X_train = [np.zeros((30, 30, 3))]
    X_train = np.asarray(X_train).reshape((len(X_train),) + X_train[0].shape + (1,))
    X_val = np.asarray(X_val).reshape((len(X_val),) + X_val[0].shape + (1,))

    y_train = classify_utils.convert_labels_to_tensors(y_train)
    y_val = classify_utils.convert_labels_to_tensors(y_val)

    # model definition
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        30, (5, 5), padding='same',
        input_shape=X_train[0].shape
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(
        60, (5, 5), padding='same'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        120, (5, 5), padding='same'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        2000
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(
        2000
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(NUMBER_OF_CLASSES,))

    model.add(tf.keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=INITIAL_ADAM_LEARNING_RATE)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # We will reinitialize the model with these
    # weights later, when we retrain the model
    # on full dataset after determining the stopping
    # times using the validation set.
    saved_initial_weights = model.get_weights()

    stopping_times = []

    for i in range(2):
        results = model.fit(

            x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=MAXIMUM_NUMBER_OF_EPOCHS, validation_data=(X_val, y_val),

            callbacks=[

                tf.keras.callbacks.EarlyStopping(

                    monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,

                    verbose=2, mode='auto'),

                Flush()]

        )

        stopping_times.append(len(results.epoch))

        print("stopped after ", stopping_times[-1], "epochs")

        # Divide the learning rate by 10

        tf.keras.backend.set_value(adam.lr, 0.1 * tf.keras.backend.get_value(adam.lr))

    # Now we will retrain the model again keeping in mind the stopping times that

    # we got by the early stopping procedure

    adam = tf.keras.optimizers.Adam(lr=INITIAL_ADAM_LEARNING_RATE)

    model.compile(

        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.set_weights(saved_initial_weights)

    for i in range(2):
        results = model.fit(

            x=np.concatenate((X_train, X_val)), y=np.concatenate((y_train, y_val)), batch_size=BATCH_SIZE, epochs=stopping_times[i],

            callbacks=[Flush()]

        )

        # Divide the learning rate by 10

        tf.keras.backend.set_value(adam.lr, 0.1 * tf.keras.backend.get_value(adam.lr))

    # model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

    model.save(os.path.join(ROOT_DIR, "classifier_subsystem/tf_models/"))

    return model


def evaluate_model(model, X_val, y_val):
    total = len(X_val)
    print(total)
    correct = 0
    for index in range(len(X_val)):
        label = model.predict(X_val[index].reshape((1,) + X_val[0].shape + (1,)))
        label = classify_utils.tensor_to_kanji(label)
        if label == y_val[index]:
            correct += 1
        # else:
        #     print("wrong prediction")
        #     print(label)
        #     print(y_val[index])
        # cv2.imshow("asdf", X_val[index])
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        print(label)
        print(y_val[index])
        print("correct" if label == y_val[index] else "wrong")
        print("\n")
    print("evaluated on {} images, labelled {} correctly.".format(total, correct))


def load_model():
    return tf.keras.models.load_model(os.path.join(ROOT_DIR, "classifier_subsystem/tf_models/"))


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = classify_utils.load_train_data()
    model = train_model( X_train, X_val, y_train, y_val)
    # model = load_model()
    evaluate_model(model, X_val, y_val)
