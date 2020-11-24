import numpy as np
from utils import extract_column, results_to_csv, error_rate
from decision_tree_starter import DecisionTree, RandomForest
from titanic_utils import preprocess_titanic, load_titanic_data, gen_maps

#RandomForest(trees, sample_size, bag_size, type_map, categories_map, seed)
#   fit(data, max_depth, min_samples)
#
#DecisionTree(type_map, categories_map)
#   fit(data, max_depth, min_samples, bag_size = None)
#
#100, 500, 70 .. 20 -> depth 36: .031884 or 26: .03478
# classifier = RandomForest(100, 500, 70, type_map, categories_map, 20)

def q_2_6():
    data, test_data, feature_names, class_names = load_titanic_data()
    data = preprocess_titanic(data, True)

    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:800], data[800:]

    type_map, categories_map = gen_maps(data)
    classifier = DecisionTree(type_map, categories_map, feature_names, class_names)
    classifier.fit(data, 3, 10)
    print(classifier)

def kaggle():
    data, test_data, feature_names, class_names = load_titanic_data()
    data = preprocess_titanic(data, True)
    test = preprocess_titanic(test_data, False)

    type_map, categories_map = gen_maps(data)
    classifier = DecisionTree(type_map, categories_map)

    classifier.fit(data, 4, 10)
    predictions = classifier.predict(test)
    pred_train = classifier.predict(data)
    actual = extract_column(data, 9)
    print(error_rate(pred_train, actual))
    results_to_csv(predictions.flatten())
    """

    data, test_data, feature_names, class_names = load_titanic_data()
    data = preprocess_titanic(data, True)

    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:800], data[800:]

    type_map, categories_map = gen_maps(data)
    classifier = DecisionTree(type_map, categories_map)

    #TESTING FOR BEST RANDOM FOREST
    best_i = -1
    best_error = 1
    best_j = -1
    print("Bagging, depth, error")
    for i in range(1, 9):
        for j in range(1,20,1):
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
    # best recorded is 2 10 at error .165829 (300 trees 300 samples)
    # best recorded is 2 5 at error .175879 (300 trees 300 samples)

    #TESTING FOR BEST DECISION TREE
    best_i = -1
    print("depth, error")
    for i in range(1, 40):
        classifier = DecisionTree(type_map, categories_map,feature_names, class_names)
        classifier.fit(data, i, 10)
        predictions = classifier.predict(valid)
        actual = extract_column(valid, 9)
        error = error_rate(predictions, actual)
        print(i, j, error)
        if error < best_error:
            best_error = error
            best_i = i
    print(best_i, best_error)
    #best recorded at 4 at point .1758
    """
def q_2_4():
    print("******RUNNING TITANIC DATA SET*****")

    data, test_data, feature_names, class_names = load_titanic_data()
    data = preprocess_titanic(data, True)

    perm = np.random.RandomState(seed=20).permutation((data.shape[0]))
    data = data[perm]
    data, valid = data[:800], data[800:]
    idy = data.shape[1] - 1

    type_map, categories_map = gen_maps(data)
    classifier = DecisionTree(type_map, categories_map)
    classifier.fit(data, 4, 10)
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


if __name__ == "__main__":
    #q_2_6()
    #kaggle()
    q_2_4()