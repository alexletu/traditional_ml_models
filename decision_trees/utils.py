import numpy as np
import pandas
from collections import Counter
import matplotlib.pyplot as plt

def results_to_csv(predictions):
    predictions = predictions.astype(int)
    df = pandas.DataFrame({'Category': predictions})
    df.index += 1
    df.to_csv('submission.csv', index_label='Id')

def sparse_to_np(sparse):
    temp = []
    for samp in range(sparse.shape[0]):
        row = sparse[samp].toarray()[0]
        temp.append(row)
    return np.asarray(temp)

def majority(data):
    idy = data.shape[1] - 1
    y = extract_column(data, idy)
    cnt = Counter(y.flatten())
    return cnt.most_common(1)[0][0]

def error_rate(prediction, actual):
    prediction = np.array(prediction.flatten())
    actual = np.array(actual.flatten())
    return np.count_nonzero(prediction - actual) / prediction.shape[0]


def extract_column(data, col):
    """
    Extracts col column from data matrix
    Outputs: a 2d array (column vector)
    """
    return data[:, [col]]


def find_index(val, thresholds):
    i = 0
    while i < len(thresholds):
        threshold = thresholds[i]
        if val < threshold:
            return i
        i += 1
    return i

def plot_data(depths, training, validation, clr_tr, clr_vld, title):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.set_dpi(100)
    plt.subplot(1, 1, 1)
    plt.plot(depths, training, label="training", color=clr_tr, marker='.', linestyle='dashed',linewidth=1, markersize=1)
    plt.plot(depths, validation, label="validation", color=clr_vld, marker='.', linestyle='dashed',linewidth=1, markersize=1)
    plt.legend()
    plt.title(title + " vs max depth")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.show()