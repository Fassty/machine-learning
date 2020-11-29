#!/usr/bin/env python
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column "class")
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data = dataset.drop("class", axis=1)
        self.target = np.array([Dataset.CLASSES.index(target) for target in dataset["class"]], np.int32)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.

    train_data, test_data, train_target, test_target = train_test_split(train.data, train.target, test_size=0.1)

    estimators = [
            ('rf', RFC(n_estimators=100, verbose=True)),
            ('svm', make_pipeline(StandardScaler(), LinearSVC(verbose=True))),
            ('bc', BaggingClassifier(verbose=True))
                ]

    model = StackingClassifier(estimators=estimators, final_estimator=GBC(n_estimators=500, max_depth=20, verbose=True))
    model.fit(train_data, train_target)

    print(f"Train set score: {accuracy_score(model.predict(train_data), train_target)}")
    print(f"Train set score: {accuracy_score(model.predict(test_data), test_target)}")

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
    # The `data` is a pandas.DataFrame containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    return model.predict(data)
