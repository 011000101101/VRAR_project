"""
KNN Model using scikit-learn implementation
"""

from sklearn import neighbors

import classifier_subsystem.model
import utils.classify_util as classify_utils


if __name__ == "__main__":
    # load data
    X_train, X_val, y_train, y_val = classify_utils.load_train_data(clean=True)

    # define model
    n_neighbors = 15
    model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

    # train
    classifier_subsystem.model.train(model, X_train, y_train, "knn")

    # evaluate
    total, correct = classifier_subsystem.model.evaluate(model, X_val, y_val, False)
    print("evaluated on {} images, labelled {} correctly.".format(total, correct))
