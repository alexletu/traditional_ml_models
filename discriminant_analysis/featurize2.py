'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

import glob
import scipy.io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_fit_matrix(filenames, vectorizer):
    print("Fitting and transforming training data")
    all_text = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            all_text.append(text)

    design_matrix = vectorizer.fit_transform(all_text)
    return design_matrix

def generate_design_matrix(test_filenames, vectorizer):
    print("Tranforming test data")
    all_text = []
    for filename in test_filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            all_text.append(text)
    return vectorizer.transform(all_text)

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW
vectorizer = TfidfVectorizer(max_features=5000, norm='l2', sublinear_tf=True)

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
X = generate_design_fit_matrix(spam_filenames + ham_filenames, vectorizer)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames, vectorizer)


Y = np.array([1]*len(spam_filenames) + [0]*len(ham_filenames)).reshape((-1, 1))

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)
