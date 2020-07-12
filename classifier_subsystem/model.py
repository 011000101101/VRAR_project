import numpy as np
import os
import cv2
import pickle
from sklearn import neighbors
import os
from utils.params import *
from utils.image_augmenting import add_noise_greyscale


def train(model, X, y, name: str):
    """
    train a model on the given training set and optionally save it to disk
    :param model: the model to train
    :param X: the sample images, list of numpy arrays (greyscale images)
    :param y: the target labels, list of strings (kanji)
    :param name: name of the model used to save it on disk, or None if it is not to be saved
    :return: the trained model
    """

    # reshape X to 2d
    X = np.asarray(X)
    X = X.reshape((X.shape[0], -1))

    print("fitting on {} samples".format(len(y)))

    # train the model
    print("begin fitting")
    model.fit(X, y)
    print("done fitting")

    # optionally save trained model
    if name is not None:
        with open("trained_{}.pkl".format(name), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    return model


def evaluate(model_or_name, X, y, interactive: bool = False):
    """
    evaluate a model on the given evaluation set
    :param model_or_name: either the model itself, or the name of a pretrained and saved model as string
    :param X: the sample images, list of numpy arrays (greyscale images)
    :param y: the target labels, list of strings (kanji)
    :param interactive: flag to visually explore the evaluation set and the predictions made
    :return: performance score as number of samples given and number of correct predictions made
    """

    # get model
    if isinstance(model_or_name, str):  # name of model passed...
        # load from disk
        with open("trained_{}.pkl".format(model_or_name), 'rb') as f:
            model = pickle.load(f)
    else:  # model passed directly...
        model = model_or_name  # use

    correct = 0

    print("evaluating on {} samples...".format(len(X)))
    # for each sample image in evaluation set
    for index in range(len(X)):
        # predict the kanji it depicts
        Z = model.predict(X[index].reshape((1, -1)))
        # count correct predictions
        if Z[0] == y[index]:
            correct += 1

        # interactively explore sample images and predictions
        if interactive:
            cv2.imshow("asdf", X[index]);cv2.waitKey();cv2.destroyAllWindows()
            print(Z)
            print(y[index])
            print("correct" if Z[0] == y[index] else "wrong")
            print("\n")
    print("finished evaluating.")

    # return total number of predictions made and number of correct predictions
    return len(X), correct


def predict(model, X: np.ndarray) -> str:
    """
    predicts the kanji for a single image
    :param model: the classifier model
    :param X: the sample image, 2d numpy array (greyscale)
    :return: the redicted label/kanji
    """
    return model.predict(X.reshape((1, -1)))
