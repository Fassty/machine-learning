#!/usr/bin/env python
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import datetime as dt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import numpy as np

class Dataset:
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        dataset = np.load(name)
        self.data, self.target = dataset["data"].reshape([-1, 28*28]).astype(np.float), dataset["target"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def grid_search():
    params_grid = dict(C=list(range(1,6)), gamma=np.outer(np.logspace(-3,3,7), [1,2,5]).flatten())
    estimator = svm.SVC()

    grid_search = GridSearchCV(estimator=estimator, param_grid=params_grid, n_jobs=-1)

    grid_result = grid_search.fit(X_train, Y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} with estimator: {param}")

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("Loading dataset")
    train = Dataset()

    # Normalize the data
    data = train.data/255.0

    print("Splitting the data")
    X_train, X_test, Y_train, Y_test = train_test_split(data, train.target,\
            test_size=0.15, random_state=args.seed)

    # grid_search()

    print("Training the model")
    model = svm.SVC(C=5, gamma=0.05, verbose = True)

    print("Fitting")
    model.fit(X_train, Y_train)
    print("Done Fitting")

    print("Predicting")
    pred = model.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(Y_test, pred) * 100} %")

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a numpy arrap containt test set input.

    args = parser.parse_args([])
    data = data/255.0
X_train, X_test, Y_train, Y_test
    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    return np.array(model.predict(data))
