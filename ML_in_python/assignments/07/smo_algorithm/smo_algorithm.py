#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float, help="Inverse regularization strenth")
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--num_passes", default=10, type=int, help="Number of passes without changes to stop after")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_ratio, random_state=args.seed)

    # We consider the following kernels:
    # - linear: K(x, y) = x^T y
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))

    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    j_generator = np.random.RandomState(args.seed)

    K_dim = range(train_data.shape[0])

    K = [[kernel(train_data[i], train_data[j]) for j in K_dim] for i in K_dim]

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


        # TODO: After each iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        def predict(X):
            pred = sum(a[j] * train_target[j] * kernel(train_data[j],X) for j in range(len(a))) + b
            if pred > -args.tolerance:
                return 1
            else:
                return -1


        accuracy = lambda X, t: np.mean([1 if predict(X[i]) == t[i] else 0 for i in range(X.shape[0])])
        train_accuracy = accuracy(train_data, train_target)
        test_accuracy = accuracy(test_data, test_target)

        print("Train acc {:.1f}%, test acc {:.1f}%".format(
            100 * train_accuracy,
            100 * test_accuracy,
        ))

    if args.plot:
        def predict(x):
            return sum(a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))) + b
        xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
        ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
        predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
        plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
        plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
        plt.scatter(train_data[a > args.tolerance, 0], train_data[a > args.tolerance, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
        plt.legend(loc="upper center", ncol=3)
        plt.show()
