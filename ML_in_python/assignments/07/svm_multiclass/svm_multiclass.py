#!/usr/bin/env python
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

def kernel(x, y):
    if args.kernel == "linear":
        return x @ y
    if args.kernel == "poly":
        return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
    if args.kernel == "rbf":
        return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))

def smo(train_data, train_target, test_data, args):
    # TODO: Use exactly the SMO algorithm from `smo_algorithm` assignment.
    #
    # The `j_generator` should be created every time with the same seed.

    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    j_generator = np.random.RandomState(args.seed)

    K_dim = range(train_data.shape[0])

    K = [[kernel(train_data[i], train_data[j]) for j in K_dim] for i in K_dim]

    def predict(x):
        pred = sum(a[j] * train_target[j] * kernel(train_data[j], x) for j in range(len(a))) + b
        return 1 if pred > -args.tolerance else -1

    passes = 0
    while passes < args.num_passes:
        a_changed = 0
        for i in range(len(a)):
            E_i = sum(a[j] * train_target[j] * K[i][j] for j in range(len(a))) + b - train_target[i]

            if (a[i] < args.C and train_target[i] * E_i < -args.tolerance) \
                    or (a[i] > 0 and train_target[i] * E_i > args.tolerance):
                j = j_generator.randint(len(a) - 1)
                j = j + (j >= i)

                E_j = sum(a[l] * train_target[l] * K[j][l] for l in range(len(a))) + b - train_target[j]
                dL = train_target[j] * (E_i - E_j)
                ddL = 2 * K[i][j] - K[i][i] - K[j][j]

                if ddL >= -args.tolerance:
                    continue

                aj_new = a[j] - (dL / ddL)

                if train_target[i] == train_target[j]:
                    L = max(0, a[i] + a[j] - args.C)
                    H = min(args.C, a[i] + a[j])
                else:
                    L = max(0, a[j] - a[i])
                    H = min(args.C, args.C + a[j] - a[i])

                if H - L < args.tolerance:
                    continue

                aj_new = max(L, min(aj_new, H))

                if abs(a[j] - aj_new) <= args.tolerance:
                    continue

                ai_new = a[i] - train_target[i] * train_target[j] * (aj_new - a[j])

                b_j = b - E_j - train_target[i]*(ai_new - a[i])*K[i][j] - train_target[j]*(aj_new - a[j])*K[j][j]
                b_i = b - E_i - train_target[i]*(ai_new - a[i])*K[i][i] - train_target[j]*(aj_new - a[j])*K[j][i]

                a[j], a[i] = aj_new, ai_new

                if 0 < a[i] and a[i] < args.C:
                    b = b_i
                elif 0 < a[j] and a[j] < args.C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2

                a_changed += 1

        passes = 0 if a_changed else passes + 1
    return np.array([predict(x) for x in test_data])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float, help="Inverse regularization strenth")
    parser.add_argument("--classes", default=5, type=int, help="Number of classes")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--num_passes", default=10, type=int, help="Number of passes without changes to stop after")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=701, type=int, help="Test set size")
    parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the digits dataset with specified number of classes, and normalize it.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data /= np.max(data)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes.
    scores = np.zeros(shape=(test_target.shape[0], args.classes))
    for i in range(args.classes):
        for j in range(i + 1, args.classes):
            data = np.concatenate((train_data[train_target == i], train_data[train_target == j]))
            target = np.concatenate((train_target[train_target == i], train_target[train_target == j]))
            target = np.array([1 if x == i else -1 for x in target])

            pred = smo(data, target, test_data, args)
            for k in range(len(pred)):
                if pred[k] > 0:
                    scores[k, i] += 1
                else:
                    scores[k, j] += 1
    # Then, classify the test set by majority voting, using the lowest class
    # index in case of ties. Finally compute `test accuracy`.

    test_accuracy = np.mean([1 if np.argmax(scores[i]) == test_target[i] else 0 for i in range(len(test_target))])

    print("{:.2f}".format(100 * test_accuracy))
