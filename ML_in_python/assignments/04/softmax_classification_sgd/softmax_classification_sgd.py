#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import scipy.sparse

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def accuracy(weights, data, target):
    acc = sum([np.argmax(weights.T @ data[i]) == target[i] for i in range(data.shape[0])])
    return acc / data.shape[0]

def gradient(w, X, Y):
    grad = np.zeros(shape=w.T.shape)
    for i in range(X.shape[0]):
        sm = softmax(X[i] @ w)
        sm[Y[i]] -= 1
        grad += np.outer(sm, X[i])

    return grad / args.batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = np.random.uniform(size=[train_data.shape[1], args.classes])

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])
        batches = np.array_split(permutation, len(permutation) / args.batch_size)
        dim = weights.T.shape

        X = train_data[permutation]
        Y = train_target[permutation]
        for batch in batches:
            X, Y = train_data[batch], train_target[batch]
            grad = gradient(weights, X, Y)
            weights -= args.learning_rate * grad.T

        train_acc = accuracy(weights, train_data, train_target)
        test_acc = accuracy(weights, test_data, test_target)

        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * train_acc,
            100 * test_acc
        ))
