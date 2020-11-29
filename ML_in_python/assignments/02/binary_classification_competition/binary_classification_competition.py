#!/usr/bin/env python
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class Dataset:
    def __init__(self,
                 name="binary_classification_competition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column Target)
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data, self.target = dataset.drop("Target", axis=1), dataset["Target"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="binary_classification_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a pandas.DataFrame containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    cf = data.select_dtypes(include=['object'])
    cf = cf.apply(lambda x : pd.factorize(x, sort=True)[0])
    data.update(cf)
    raw_data = data
    cat = [1, 3, 5, 6, 7, 8, 9, 13]
    nc = list(range(data.shape[1]))
    nc = list(set(nc) - set(cat))
    ct = ColumnTransformer([("Non-cat", StandardScaler(), nc),("Cat", OneHotEncoder(), cat)]).fit(raw_data)

    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    model = None
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    p_d = ct.transform(raw_data)
    return np.array(model.predict(p_d))
