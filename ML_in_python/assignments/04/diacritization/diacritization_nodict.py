#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np

import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.pipeline

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    TARGETS = 3
    @staticmethod
    def letter_to_target(letter):
        if letter in "áéíóúý":
            return 1
        if letter in "čďěňřšťůž":
            return 2
        return 0
    @staticmethod
    def target_to_letter(target, letter):
        if target == 1:
            index = "aeiouy".find(letter)
            return "áéíóúý"[index] if index >= 0 else letter
        if target == 2:
            index = "cdenrstuz".find(letter)
            return "čďěňřšťůž"[index] if index >= 0 else letter
        return letter

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.data = dataset_file.read()

    @staticmethod
    def get_features(data, args):
        processed = data.lower().translate(Dataset.DIA_TO_NODIA)
        features, targets, indices = [], [], []
        for i in range(len(processed)):
            if processed[i] not in Dataset.LETTERS_NODIA:
                continue
            features.append([])
            for o in range(-args.window_chars, args.window_chars + 1):
                features[-1].append(processed[i + o] if i + o >= 0 and i + o < len(processed) else "<pad>")
            for s in range(1, args.window_ngrams):
                for o in range(-s, 0+1):
                    features[-1].append(processed[max(i+o, 0):i+o+s+1])
            targets.append(Dataset.letter_to_target(data[i].lower()))
            indices.append(i)

        return features, targets, indices


parser = argparse.ArgumentParser()
parser.add_argument("--estimator", default="lr", type=str, help="Estimator to use")
parser.add_argument("--hidden_layers", nargs="+", default=[100], type=int, help="Hidden layer sizes")
parser.add_argument("--max_iter", default=100, type=int, help="Max iters")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--size", default=None, type=int, help="Train size to use")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--window_chars", default=3, type=int, help="Window characters to use")
parser.add_argument("--window_ngrams", default=5, type=int, help="Window ngrams to use")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.
    train_data, train_targets, _ = Dataset.get_features(train.data, args)

    model = sklearn.pipeline.Pipeline([
        ("one-hot", sklearn.preprocessing.OneHotEncoder(categories="auto", handle_unknown="ignore")),
        ("estimator", {
            "lr": sklearn.linear_model.LogisticRegression(solver=args.solver, multi_class="multinomial", max_iter=args.max_iter, verbose=1),
            "nn": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layers, max_iter=args.max_iter, verbose=1)}[args.estimator]),
    ])
    model.fit(train_data[:args.size], train_targets[:args.size])

    # Compress the model
    for estimator in [model["estimator"]]:
        if hasattr(estimator, "_optimizer"): estimator._optimizer = None
        for attr in ["coef_", "intercept_"]:
            if hasattr(estimator, attr): setattr(estimator, attr, getattr(estimator, attr).astype(np.float16))
        for attr in ["coefs_", "intercepts_"]:
            if hasattr(estimator, attr): setattr(estimator, attr, [c.astype(np.float16) for c in getattr(estimator, attr)])

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path + "{}{}-{}-{}-{}".format(args.estimator, args.hidden_layers[0], args.window_chars, args.window_ngrams, args.max_iter), "wb") as model_file:
        pickle.dump((model, args), model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a `str` containing text without diacritics
    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model, args = pickle.load(model_file)

    # TODO: Return the predictions as a diacritized `str`. It has to have
    # exactly the same length as `data`.
    test_data, _, test_indices = Dataset.get_features(data, args)
    test_targets = model.predict(test_data)
    predictions = list(data)
    for i in range(len(test_targets)):
        predictions[test_indices[i]] = Dataset.target_to_letter(test_targets[i], data[test_indices[i]].lower())
        if data[test_indices[i]].isupper():
            predictions[test_indices[i]] = predictions[test_indices[i]].upper()
    return "".join(predictions)
