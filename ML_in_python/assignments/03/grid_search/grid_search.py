#!/usr/bin/env python
import argparse
import sys

import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    print(dataset.DESCR, file=sys.stderr)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_ratio, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(multi_class="multinomial")
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbgfs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.
    pipeline = Pipeline([
        ("Scaler", MinMaxScaler()),
        ("Polynomial", PolynomialFeatures()),
        ("LogRe", LogisticRegression(multi_class="multinomial"))])

    kfold = StratifiedKFold(5)
    params_grid = dict(Polynomial__degree=[1,2],LogRe__C=[0.01,1,100], LogRe__solver=["lbfgs", "sag"])
    grid = GridSearchCV(estimator=pipeline, param_grid=params_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(train_data, train_target)

    model = grid_result.best_estimator_
    pred = model.predict(test_data)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, pred)

    print("{:.2f}".format(100 * test_accuracy))
