from utils import extract_column
from sklearn.impute import SimpleImputer
import numpy as np
import re
import pandas

def preprocess_titanic(data, include_labels):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    columns = [0 for _ in range(9)]

    current_col = extract_column(data, 5).flatten()
    current_col = [re.sub('[^0-9]', '', ticket) for ticket in current_col]                      #remove all letters
    current_col = [float(ticket) if ticket != '' else float(np.nan) for ticket in current_col]  #convert to float
    current_col = np.array([current_col]).T
    current_col = imp_mean.fit_transform(current_col)
    columns[5] = current_col

    for col in [0, 1, 8]:
        current_col = extract_column(data, col)
        current_col =imp_mode.fit_transform(current_col)
        columns[col] = (current_col)
    for col in [2, 3, 4, 6]:
        current_col = extract_column(data, col)
        current_col = imp_mean.fit_transform(current_col)
        columns[col] = (current_col)

    current_col = extract_column(data, 7).flatten()
    current_col = [re.sub('[^A-z]', '', cabin)[0] if isinstance(cabin, str) else cabin for cabin in current_col]
    class_col = extract_column(data, 0).flatten()
    current_col = ['B' if not isinstance(current_col[i], str) and class_col[i] == 1 else current_col[i] for i in
                   range(len(current_col))]
    current_col = ['D' if not isinstance(current_col[i], str) and class_col[i] == 2 else current_col[i] for i in
                   range(len(current_col))]
    current_col = ['F' if not isinstance(current_col[i], str) and class_col[i] == 3 else current_col[i] for i in
                   range(len(current_col))]
    current_col = np.array([current_col]).T
    columns[7] = current_col

    if include_labels:
        columns.append(extract_column(data, 9))
    return np.hstack(tuple(columns))

def load_titanic_data():
    path_train = 'datasets/titanic/titanic_training.csv'
    data = pandas.read_csv(path_train, delimiter=',', dtype=None)
    path_test = 'datasets/titanic/titanic_testing_data.csv'
    test_data = pandas.read_csv(path_test, delimiter=',', dtype=None)
    data = data[data.survived >= 0]
    y = data.values[:, [0]]  # label = survived
    X = data.values[:, range(1, 10)]
    header = list(data.columns)
    header.append(header.pop(0))
    #print(header)
    data = np.hstack((X, y))
    #print(data.shape)
    #print(test_data.shape)
    test_data = test_data.values
    class_names =['died', 'survived']
    return data, test_data, header, class_names

def gen_maps(data):
    type_map = {}
    categories_map = {}

    for i in [0, 1, 7, 8]:
        type_map[i] = 'categorical'
    for i in [2, 3, 4, 6]:
        type_map[i] = 'quantitative'
    type_map[5] = 'clustered'

    categories_map[0] = [1, 2, 3]
    categories_map[1] = ['male', 'female']
    categories_map[7] = list(set(extract_column(data, 7).flatten()))
    categories_map[8] = ['S', 'C', 'Q']
    return type_map, categories_map