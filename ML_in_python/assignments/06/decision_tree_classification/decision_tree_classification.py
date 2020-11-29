#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

class Tree:
    def __init__(self):
        self.c_func = self.c_gini if args.criterion == 'gini' else self.c_entropy
        self.root = None
        self.args = args
        self.classes = None
        self.splits = 0

    def fit(self, X, t):
        self.classes = list(set(t))
        self.root = Node(self, np.array(list(range(X.shape[0]))))
        self.root.find_best_split()
        self.splits += 1

        if args.max_leaves is None:
            self.root = self.split_recursive(self.root)
        else:
            self.root = self.split_iterative(self.root)

    def c_gini(self, indices):
        p = np.zeros(shape=len(self.classes))
        for (i, k) in enumerate(self.classes):
            uniq, count = np.unique(train_target[indices], return_counts=True)
            try:
                p[i] = dict(zip(uniq, count))[k] / len(indices)
            except:
                p[i] = 0

        return len(indices) * sum(p[i] * (1 - p[i]) for i in range(len(self.classes)))

    def c_entropy(self, indices):
        p = np.zeros(len(self.classes))
        for (i, k) in enumerate(self.classes):
            uniq, count = np.unique(train_target[indices], return_counts=True)
            try:
                p[i] = dict(zip(uniq, count))[k] / len(indices)
            except:
                p[i] = 0
        return -len(indices) * sum(p[i] * np.log(p[i]) if p[i] != 0 else 0 for i in range(len(self.classes)))

    def split_iterative(self, node):
        # We will be iteratively creating the tree, so the root node will be split first
        # and then always the one with the largest difference of c-scores
        leaves = []
        leaves.append(node)
        for i in range(args.max_leaves - 1):
            best_leaf = None

            # Find the leaf with largest difference
            for leaf in leaves:
                if leaf.size >= args.min_to_split\
                and (best_leaf is None or best_leaf.diff < leaf.diff):
                    best_leaf = leaf

            # Split the best leaf(on the first iteration this will be the root node so I don't have to
            # calculate the threshold value and can just split)
            left, right = best_leaf.split()

            # Calculate threshold and split feature
            left.find_best_split()
            right.find_best_split()

            # Add the two newly created leaves to the leaf list and remove their parent
            leaves.append(left)
            leaves.append(right)
            leaves.remove(best_leaf)

        # Return back the root node
        return node

    def split_recursive(self, node):
        # Recursively build the tree drom left to right
        if node.size >= args.min_to_split\
        and node.c != 0\
        and (args.max_depth is None or node.depth < args.max_depth):
            left, right = node.split()
            left.find_best_split()
            right.find_best_split()
            self.split_recursive(left)
            self.split_recursive(right)

        # Return back the root node
        return node

    def predict(self, X):
        node = self.root
        # Travel the tree from root to the right leaf
        while node.left is not None:
            if X[node.split_feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        targets = train_target[node.indices]
        class_distrib = np.zeros(shape=len(self.classes))
        for (i, k) in enumerate(self.classes):
            uniq, count = np.unique(targets, return_counts=True)
            try:
                class_distrib[i] = dict(zip(uniq,count))[k]
            except:
                class_distrib[i] = 0

        return np.argmax(class_distrib)

class Node:
    def __init__(self, tree, indices, depth=0):
        self.tree = tree
        self.size = train_data[indices].shape[0]
        self.indices = indices
        self.diff = 0
        self.threshold = None
        self.split_feature = None
        self.c = None
        self.depth = depth
        self.left = None
        self.right = None


    def find_best_split(self):
        self.c = self.tree.c_func(self.indices)

        # Best split params
        best_c, best_feature, best_threshold = 0, 0, 0

        # Iterate features in sequential order
        for feature in range(train_data.shape[1]):
            # Need to sort the data in ascending order and retrieve indices
            X = train_data[self.indices]
            sorted_ids = self.indices[np.argsort(X[:, feature])]
            row = [self.indices, feature]

            # Try to choose each value as a threshold and calculate the information gain
            for i in range(len(sorted_ids)):
                threshold = train_data[sorted_ids[i], feature]

                # Split the indices by threshold
                l_ids, r_ids = self.indices[train_data[row] <= threshold],\
                        self.indices[train_data[row] > threshold]

                c_r = self.tree.c_func(l_ids)
                c_l = self.tree.c_func(r_ids)
                c = self.c - c_r - c_l

                if c > best_c:
                    best_c = c
                    best_feature = feature
                    best_threshold = threshold

            self.diff = best_c
            self.split_feature = best_feature
            self.threshold = best_threshold

    def split(self):
        if self.split_feature is None:
            self.find_best_split()

        l_ids, r_ids = self.indices[train_data[self.indices, self.split_feature] <= self.threshold],\
                self.indices[train_data[self.indices, self.split_feature] > self.threshold]

        self.left = Node(self.tree, l_ids, depth=self.depth + 1)
        self.right = Node(self.tree, r_ids, depth=self.depth + 1)

        return self.left, self.right

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
    parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
    parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
    parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=42, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    # smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest feature values
    #   of the instances corresponding to the given node (i.e., for three instances
    #   with values 1, 7, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be at most `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    tree = Tree()

    tree.fit(train_data, train_target)

    accuracy = lambda data, target: np.mean([1 if tree.predict(data[i]) == target[i] else 0 for i in range(data.shape[0])])

    train_accuracy = accuracy(train_data, train_target)
    test_accuracy = accuracy(test_data, test_target)

    # TODO: Finally, measure the training and testing accuracy.
    print("Train acc: {:.1f}%".format(100 * train_accuracy))
    print("Test acc: {:.1f}%".format(100 * test_accuracy))
