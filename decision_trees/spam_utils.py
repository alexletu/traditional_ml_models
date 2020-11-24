from utils import extract_column
from sklearn.impute import SimpleImputer
import numpy as np
import re
import pandas
import scipy.io
from utils import sparse_to_np

def load_spam():
    path_train = 'datasets/spam-dataset/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    X = sparse_to_np(X)
    y = data['training_labels']
    Z = data['test_data']
    Z = sparse_to_np(Z)
    #feature_names = data['feature_names']
    feature_names = []
    class_names = ["Ham", "Spam"]
    #print(X.shape)
    #print(y.shape)
    #print(Z.shape)
    data = np.hstack((X, y))

    feature_path = 'datasets/spam-dataset/feature_names.mat'
    feature_dict = scipy.io.loadmat(feature_path)
    feature_names = feature_dict['feature_names']
    return data, Z, feature_names, class_names

