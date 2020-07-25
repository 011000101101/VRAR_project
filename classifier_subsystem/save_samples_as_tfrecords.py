import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.model_selection import train_test_split

from utils.params import *
import utils.classify_util as classify_utils

# load training data
with open("../bin_blobs/kanji_image_samples_augmentes.pkl", 'rb') as f:
    image_samples_augmented = pickle.load(f)

X = [sample[0] for sample in image_samples_augmented]
y = [sample[1] for sample in image_samples_augmented]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, shuffle=True)

print(len(X), len(X_train), len(X_val))

X = [X_train, X_val]
y = [y_train, y_val]
subfolder = ["train/", "val/"]

# for i in range(2):
#
#     item_counter_in_record = 0
#     record_counter = 0
#     current_base_path = os.path.join(os.path.join(ROOT_DIR, "classifier_subsystem/tfrecords/"), subfolder[i])
#     if not os.path.isdir(current_base_path):
#         os.mkdir(current_base_path)
#     writer = tf.io.TFRecordWriter(os.path.join(current_base_path + "000.tfrecord"))
#
#     for sample, label in zip(X[i], y[i]):
#
#         item_counter_in_record += 1
#
#         if item_counter_in_record > 1000:  # "the recommended number of images stored in one tfrecord file is 1000."
#             item_counter_in_record = 1
#             record_counter += 1
#             writer.close()
#             writer = tf.io.TFRecordWriter(current_base_path + "%.3d.tfrecord" % record_counter)
#             print("Creating the %.3d tfrecord file" % record_counter)
#
#         img_raw = sample.flatten()
#         label_raw = classify_utils.convert_label_to_tensor(label)
#
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "mapped_data": tf.train.Feature(int64_list=tf.train.Int64List(value=img_raw)),
#             "result": tf.train.Feature(int64_list=tf.train.Int64List(value=label_raw))}))
#         writer.write(example.SerializeToString())
#
#     writer.close()