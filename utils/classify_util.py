import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils.params import *

index_dict = dict()

with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_list.pkl"), 'rb') as f:
    kanji_list = pickle.load(f)
for i in range(len(kanji_list)):
    index_dict[kanji_list[i]] = i
kanji_indices = np.asarray(kanji_list)
N_VALUES = len(kanji_list)


def load_train_data(clean: bool = True):
    """
    loads the images and respective kanji labels from the disk
    :return: list of samples (X), and list of respective kanji labels (y)
    """
    # load training data
    with open("../bin_blobs/kanji_image_samples_augmentes.pkl", 'rb') as f:
        image_samples_augmented = pickle.load(f)

    # extract input (X) and expected output (y) for each sample
    X = [sample[0] for sample in image_samples_augmented]
    y = [sample[1] for sample in image_samples_augmented]

    # optionally limit number of samples for faster training
    X = X[:100000]
    y = y[:100000]

    # split into training and evaluation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, shuffle=True)

    if clean:
        X_train = []
        y_train = []
        with open("../bin_blobs/kanji_image_samples.pkl", 'rb') as f:
            image_samples = pickle.load(f)
        for kanji in image_samples.keys():
            for image in image_samples[kanji]:
                X_train.append(image[1])
                y_train.append(kanji)

    return X_train, X_val, y_train, y_val


def convert_labels_to_tensors(y: list):
    indices = [index_dict[kanji] for kanji in y]
    return np.eye(N_VALUES)[indices]


def tensor_to_kanji(tensor: np.ndarray):
    label = np.argmax(tensor)
    return kanji_indices[label]
