"""
KNN Model using scikit-learn implementation
"""

import pickle
from sklearn import neighbors
from sklearn.model_selection import train_test_split

import classifier_subsystem.model


def load_train_data():
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
    # X = X[:3000]
    # y = y[:3000]

    return X, y


def train_only(X, y):
    """
    trains the knn
    :param X: sample images
    :param y: kanji labels
    :return: the trained knn model
    """
    n_neighbors = 15
    model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    classifier_subsystem.model.train(model, X, y, "knn")
    return model


def eval_only(model, X, y, interactive):
    """
    evaluate the knn and return the performance
    :param model: the trained model to evaluate
    :param X: sample images
    :param y: ground truth (kanji labels)
    :param interactive: flag to visually inspect the samples and observe the predictions
    :return: evaluation scores
    """
    return classifier_subsystem.model.evaluate(model, X, y, interactive)


if __name__ == "__main__":
    # load data
    X_train_val, y_train_val = load_train_data()
    # split into training and evaluation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.01, shuffle=True)

    # train
    # model = train_only(X_train, y_train)

    # evaluate
    total, correct = eval_only("knn", X_val, y_val, False)
    print("evaluated on {} images, labelled {} correctly.".format(total, correct))
