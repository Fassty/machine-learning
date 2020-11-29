#!/usr/bin/env python
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer as CoV
from sklearn.feature_extraction.text import TfidfVectorizer as TFV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer as TFT
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

import sklearn.metrics

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(name.replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a Python list containing tweets as `str`ings.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)


    # TODO: Return the predictions as a Python list or Numpy array of
    # binary labels of the tweets.

    pred = np.array(model.predict(data))
    return pred

class Model:
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('bow', CoV(ngram_range=(2,3))),
         #  ('tfidf', TFT()),
            ('naive bayes', GBC(n_estimators=10))
            ])

    def fit(self, data, target):
        self.pipeline.fit(data, target)

    def predict(self, data):
        return self.pipeline.predict(data)

    def score(self, data, target):
        pred = self.predict(data)
        return f1_score(pred, target)

if __name__ == "__main__":

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.

    train_data, test_data, train_target, test_target = train_test_split(train.data, train.target, test_size=0.1)

    estimators = [
        ('gbc', GBC(n_estimators=10)),
        ('bc', MNB()),
        ('lr', LogisticRegression())
            ]

    model = Pipeline([
        ('bow', CoV(ngram_range=(2,3))),
        #('tfidf', TFT()),
        ('gbc', GBC(n_estimators=10))
        ])

    model.fit(train.data, train.target)

    print(f"Training set score: {f1_score(model.predict(train_data), train_target):.3f}")
    print(f"Test set score: {f1_score(model.predict(test_data), test_target):.3f}")

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    recodex_predict(test_data)

