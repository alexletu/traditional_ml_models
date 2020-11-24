from collections import Counter
import numpy as np
import random
from utils import extract_column, majority, find_index

def gen_clustered_thresholds(data, idx):
    sorted_data = data[data[:, idx].argsort()]
    idy = data.shape[1] - 1
    thresholds = []
    if len(data) == 1:
        return [data[0][idx]]
    for i in range(1, len(sorted_data)):
        prev = data[i - 1]
        curr = data[i]
        prev_y = prev[idy]
        curr_y = curr[idy]
        curr_x = curr[idx]
        prev_x = prev[idx]
        if prev_y != curr_y:
            thresholds.append((curr_x + prev_x) / 2)
    return thresholds


def gen_quantitative_thresholds(data, idx):
    xs = extract_column(data, idx)
    return [np.mean(xs)]


def gen_quantitative_thresholds_exact(data, idx):
    thresholds = gen_clustered_thresholds(data, idx)

    best_purification = -1
    best_threshold = None
    for threshold in thresholds:
        purification = DecisionTree.gini_purification(data, idx, False, [threshold])
        if purification > best_purification:
            best_purification = purification
            best_threshold = threshold
    return [best_threshold] if best_threshold is not None else gen_quantitative_thresholds(data, idx)


class Node:
    """
    depth is an integer
    split_idx denotes the feature being split on 0 - d
        IF leaf node, then split_idx is None
    split_thresholds denotes:
        categories to split on -> len(children) = len(thresholds)
        quantitative data to split on -> len(children) = len(thresholds) + 1
        IF leaf node, then split_thresholds is None
    split_categorical is a boolean, denotes if this node follows categorical or quantitative splitting
        IF leaf node, then split_categorical is None
    is_leaf is a boolean denoting if this is leaf node or not
    children is a list of children
        IF leaf, children will be the empty list
    label denotes the class which this node represents
        IF not leaf, label will be None
    """
    def __init__(self, depth, feature_names = None, class_names = None):
        self.depth = depth
        self.split_idx = None
        self.split_thresholds = None
        self.split_categorical = None
        self.is_leaf = None
        self.children = []
        self.label = None
        self.feature_names = feature_names
        self.class_names = class_names

    def set_leaf(self, label):
        self.is_leaf = True
        self.label = label

    def set_internal(self, idx, categorical, thresholds, children, label):
        self.split_idx = idx
        self.split_categorical = categorical
        self.split_thresholds = thresholds
        self.is_leaf = False
        self.children = children
        self.label = label

    def __repr__(self):
        offset = "\n" + self.depth * "     "
        s = ""
        if self.is_leaf:
            s += offset + "Leaf Node"
            s += offset + " Depth = " + str(self.depth)
            s += offset + " Label = " + (str(self.label) if self.class_names is None else self.class_names[int(self.label)])
            return s
        else:
            s += offset + "Internal Node"
            s += offset + " Depth = " + str(self.depth)
            s += offset + " Split on feat = " + (str(self.split_idx) if self.feature_names is None else self.feature_names[self.split_idx])
            s += offset + " Categorical split = " + str(self.split_categorical)
            s += offset + " Categories/Thresholds = " + str(self.split_thresholds)
            for child in self.children:
                s += child.__repr__()
            return s
    def string(self):
        offset = "\n" + self.depth * "     "
        s = ""
        if self.is_leaf:
            s += offset + "Leaf Node"
            s += offset + " Depth = " + str(self.depth)
            s += offset + " Label = " + (str(self.label) if self.class_names is None else self.class_names[int(self.label)])
            return s
        else:
            s += offset + "Internal Node"
            s += offset + " Depth = " + str(self.depth)
            s += offset + " Split on feat = " + (str(self.split_idx) if self.feature_names is None else self.feature_names[self.split_idx])
            s += offset + " Categorical split = " + str(self.split_categorical)
            s += offset + " Categories/Thresholds = " + str(self.split_thresholds)
        return s

class DecisionTree:
    def __init__(self, type_map, categories_map, feature_names = None, class_names = None):
        self.type_map = type_map
        self.categories_map = categories_map
        self.root = None
        self.bag_size = -1
        self.feature_names = feature_names
        self.class_names = class_names

    @staticmethod
    def gini_impurity(y):
        """
        Input: 2d array of class labels (column vector)
        Output float number that represents the gini impurity. Low is good.
        """
        cnt = Counter(y.flatten())
        gini = 0
        total = len(y)
        for elem in list(cnt):  # elements of cnt are classes
            gini += (cnt[elem] / total) ** 2
        return 1 - gini


    @staticmethod
    def gini_purification(data, idx, categorical, thresholds):
        """
        Input: data (last column is label column), feature to split on, whether or not the split is categorical,
            list of thresholds or categories
        Output: the change in gini-impurity, lower (negative) is good.
        """
        idy = data.shape[1] -1
        y = extract_column(data, idy)
        x = extract_column(data, idx)
        gini_before = DecisionTree.gini_impurity(y)
        gini_after = 0

        # joined represents two columns, left is idx feature, right is the label
        joined = np.hstack((x,y))
        total = len(joined)

        if categorical:
            for category in thresholds:
                split_data = joined[joined[:, 0] == category]
                new_y = extract_column(split_data, 1)
                gini_after += DecisionTree.gini_impurity(new_y) * len(new_y)
        else:
            for thresh in thresholds:
                split_data_below = joined[joined[:, 0] < thresh]
                joined = joined[joined[:, 0] >= thresh]
                new_y = extract_column(split_data_below, 1)
                gini_after += DecisionTree.gini_impurity(new_y) * len(new_y)
            new_y = extract_column(joined, 1)
            gini_after += DecisionTree.gini_impurity(new_y) * len(new_y)
        gini_after /= total

        return gini_before - gini_after

    def split(self, data, idx, categorical, thresholds):
        """
                Input: data (last column is label column), feature to split on, whether or not the split is categorical,
                    list of thresholds or categories
                Output: a python list of numpy 2-d arrays representing the split data based on idx and thresholds
        """
        total_data = []
        if categorical:
            for category in thresholds:
                split_data = data[data[:, idx] == category]
                total_data.append(split_data)
        else:
            for thresh in thresholds:
                split_data_below = data[data[:, idx] < thresh]
                data = data[data[:, idx] >= thresh]
                total_data.append(split_data_below)
            total_data.append(data)
        return total_data
    
    def segmenter(self, data):
        """
        Input: data matrix, last column are labels
        Output returns the feature to split on (idx), categorical or not, the categories or thresholds to split on,
            the best purification
        """
        best_idx = -1
        best_categorical = None
        best_thresholds = []
        best_purification = -1
        # for each feature
        sampled_features = np.random.choice(a=list(range(0, data.shape[1] - 1)), size=self.bag_size ,replace=False)
        for idx in sampled_features:
            feature_type = self.type_map[idx]
            if feature_type == 'categorical':
                thresholds = self.categories_map[idx]                                      # these are really categories
                purification = DecisionTree.gini_purification(data, idx, True, thresholds)
                categorical = True
            elif feature_type == 'clustered':
                thresholds = gen_clustered_thresholds(data, idx)
                purification = DecisionTree.gini_purification(data, idx, False, thresholds)
                categorical = False
            else:
                #thresholds = gen_quantitative_thresholds(data, idx)                    #use with spam
                thresholds = gen_quantitative_thresholds_exact(data, idx)               #use with titanic
                purification = DecisionTree.gini_purification(data, idx, False, thresholds)
                categorical = False
            if purification > best_purification:
                best_idx = idx
                best_categorical = categorical
                best_thresholds = thresholds
                best_purification = purification
        return best_idx, best_categorical, best_thresholds, best_purification

    def fit_helper(self, data, current_depth, max_depth, min_samples, node):
        if data.shape[0] == 0:
            node.is_leaf = True
            return
        if current_depth >= max_depth or len(data) < min_samples:
            label = majority(data)
            node.set_leaf(label)
            return
        else:
            idx, categorical, thresholds, purification = self.segmenter(data)
            if purification <= .0001 or thresholds is None:
                label = majority(data)
                node.set_leaf(label)
                return
            split_data = self.split(data, idx, categorical, thresholds)
            new_children = [Node(current_depth + 1, self.feature_names, self.class_names) for _ in range(len(split_data))]
            label = majority(data)
            node.set_internal(idx, categorical, thresholds, new_children, label)
            for i in range(len(split_data)):
                node.children[i].label = label
                self.fit_helper(split_data[i], current_depth + 1, max_depth, min_samples, node.children[i])

    def fit(self, data, max_depth, min_samples, bag_size = None):
        if bag_size is None:
            self.bag_size = data.shape[1] - 1
        else:
            self.bag_size = bag_size
        self.root = Node(0, self.feature_names, self.class_names)
        self.fit_helper(data, 0, max_depth, min_samples, self.root)
        return

    def predict_helper(self, x, node, verbose):
        offset = node.depth * "     "
        if verbose:
            print(node.string())

        if node.is_leaf:
            return node.label

        idx = node.split_idx
        categorical = node.split_categorical
        thresholds = node.split_thresholds

        if categorical:
            categories = thresholds
            x_val = x[idx]
            child = categories.index(x_val)
            if verbose:
                print(offset + " Observed value: ", x_val)
            return self.predict_helper(x, node.children[child], verbose)
        else:
            x_val = x[idx]
            child = find_index(x_val, thresholds)
            if verbose:
                print(offset + " Observed value: ", x_val)
            return self.predict_helper(x, node.children[child], verbose)

    def predict(self, X, verbose = False):
        predictions = []
        for x in X:
            prediction = self.predict_helper(x, self.root, verbose)
            predictions.append(prediction)
        return np.array([predictions]).T

    def __repr__(self):
        return self.root.__repr__()


class RandomForest:
    def __init__(self, trees, sample_size, bag_size, type_map, categories_map, seed):
        self.trees = [DecisionTree(type_map, categories_map) for _ in range(trees)]
        self.sample_size = sample_size
        self.seed = seed
        self.bag_size = bag_size

    def fit(self, data, max_depth, min_samples):
        random.seed(self.seed)
        for tree in self.trees:
            sample = data[np.random.randint(data.shape[0], size=self.sample_size), :]
            tree.fit(sample, max_depth, min_samples, self.bag_size)
        return
    
    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(X))
        tree_predictions = np.hstack(tree_predictions)
        final_predictions = [Counter(ensemble).most_common(1)[0][0] for ensemble in tree_predictions]
        return np.array(final_predictions).T
