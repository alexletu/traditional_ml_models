import numpy as np
from utils import extract_column, results_to_csv, error_rate, plot_data
from spam_utils import load_spam
from decision_tree_starter import DecisionTree, RandomForest

#RandomForest(trees, sample_size, bag_size, type_map, categories_map, seed)
#   fit(data, max_depth, min_samples)
#
#DecisionTree(type_map, categories_map)
#   fit(data, max_depth, min_samples, bag_size = None)
#
#100, 500, 70 .. 20 -> depth 36: .031884 or 26: .03478
# classifier = RandomForest(100, 500, 70, type_map, categories_map, 20)


def plot_q2_5_3():
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}
    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:4137], data[4137:]

    train_accuracies = []
    valid_accuracries =[]
    depths = list(range(1,41))

    for max_depth in range(1,41):
        print("Computing max depth: ", max_depth)
        idy = valid.shape[1] - 1
        classifier = DecisionTree(type_map, categories_map, feature_names, class_names)
        classifier.fit(data, max_depth, 10)
        train_pred = classifier.predict(data)
        valid_pred = classifier.predict(valid)
        train_actual = extract_column(data, idy)
        valid_actual = extract_column(valid, idy)
        train_acc = 1 - error_rate(train_pred, train_actual)
        valid_acc = 1 -error_rate(valid_pred, valid_actual)
        train_accuracies.append(train_acc)
        valid_accuracries.append(valid_acc)

    plot_data(depths, train_accuracies, valid_accuracries, 'r', 'b', 'Training/Validation Accuracies')
    return

def q_2_5_2():
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}
    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:4137], data[4137:]

    classifier = DecisionTree(type_map, categories_map, feature_names, class_names)
    classifier.fit(data, 8, 15)
    samp_point = np.array([valid[0]])
    classifier.predict(samp_point, True)

    samp_point = np.array([valid[1]])
    classifier.predict(samp_point, True)

def kaggle():
    """
    #run featurize.py with 5000 samples
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}

    classifier = RandomForest(100, 500, 70, type_map, categories_map, 20)
    classifier.fit(data, 36, 10)
    predictions = classifier.predict(test_data)
    pred_train = classifier.predict(data)
    actual = extract_column(data, 9)
    print(error_rate(pred_train, actual))
    results_to_csv(predictions.flatten())

    #TESTING DECISION TREE
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}
    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:4137], data[4137:]

    best_i = -1
    best_error = 1
    for i in range(2, 50):
        classifier = DecisionTree(type_map, categories_map, feature_names, class_names)
        classifier.fit(data, 36, 10)
        predictions = classifier.predict(valid)
        actual = extract_column(valid, valid.shape[1] - 1)
        error = error_rate(predictions, actual)
        print(i, error)
        if error < best_error:
            best_error = error
            best_i = i
    print(best_i, best_error)
    # Best at depth 14 with error 0.11594202898550725
    """
    """
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}
    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:4137], data[4137:]

    best_i = -1
    best_error = 1
    best_j = -1
    print("Bagging, depth, error")
    for i in range(10, 50):
        for j in range(30, 31):
            classifier = RandomForest(300, 300, i, type_map, categories_map, 20)
            classifier.fit(data, j, 10)
            predictions = classifier.predict(valid)
            actual = extract_column(valid, 9)
            error = error_rate(predictions, actual)
            print(i, j, error)
            if error < best_error:
                best_error = error
                best_i = i
                best_j = j
    print(best_i, best_j, best_error)
    """
    return

def q2_4():
    print("******RUNNING SPAM DATA SET*****")
    data, test_data, feature_names, class_names = load_spam()
    type_map = dict((i, 'quantitative') for i in range(data.shape[1]))
    categories_map = {}
    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:4137], data[4137:]
    idy = data.shape[1] - 1

    classifier = DecisionTree(type_map, categories_map)
    classifier.fit(data, 14 , 10)
    train_predictions = classifier.predict(data)
    train_actual = extract_column(data, idy)
    valid_predictions = classifier.predict(valid)
    valid_actual = extract_column(valid, idy)

    print("Decision Tree training Accuracies:       ", error_rate(train_predictions, train_actual))
    print("Decision Tree Validation Accuracies:    ", error_rate(valid_predictions, valid_actual))

    classifier = RandomForest(300, 300, 2, type_map, categories_map, 20)
    classifier.fit(data, 10, 10)

    train_predictions = classifier.predict(data)
    train_actual = extract_column(data, idy)
    valid_predictions = classifier.predict(valid)
    valid_actual = extract_column(valid, idy)

    print("Random Forest training Accuracies:       ", error_rate(train_predictions, train_actual))
    print("Random Forest Validation Accuracies:    ", error_rate(valid_predictions, valid_actual))

    return

if __name__ == "__main__":
    #plot_q2_5_3()
    #q_2_5_2()
    #kaggle()
    q2_4()