#!/usr/bin/env python3
import argparse
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="linear_regression_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a Numpy array containt test set input.

    args = parser.parse_args([])
    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with open(args.model_path, "rb") as model_file:
         model = pickle.load(model_file)

    # Standardization of the dataset
    #
    # Make the data behave more like a normal distribution
    # so they would be centered around 0
    scaler = StandardScaler().fit(data)
    rescaled_data = scaler.transform(data)

    # TODO: Return the predictions as a Numpy array.
    return np.array(model.predict(rescaled_data))

