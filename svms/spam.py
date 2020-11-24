import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from sklearn.metrics import accuracy_score
from save_csv import results_to_csv

for data_name in ["spam"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)

def permute_dictionaries(training_data, training_labels):
	#takes two dictionaries and permutes both while keeping consistency
	perm = np.random.RandomState(seed=100).permutation(training_data.shape[0])
	return (training_data[perm], training_labels[perm])

spam_total_data = io.loadmat("data/%s_data.mat" % "spam")

spam_training_data = spam_total_data["training_data"]
spam_training_data_labels = spam_total_data["training_labels"]
spam_training_data, spam_training_data_labels = permute_dictionaries(spam_training_data, spam_training_data_labels)
spam_test_data = spam_total_data["test_data"]
print("train")
print(spam_training_data)
print("test")
print(spam_test_data)

print("spam_training_data", spam_training_data.shape)
print("spam_training_data_labels", spam_training_data_labels.shape)
print("spam_test_data", spam_test_data.shape)

def problem6(training_data, training_data_labels, test_data, C_Value = 0):

	classifier = svm.LinearSVC(random_state = 40, C = 10 ** C_Value)

	classifier.fit(training_data, np.ravel(training_data_labels))

	predict_training_results = classifier.predict(training_data)
	print(accuracy_score(np.ravel(training_data_labels), np.ravel(predict_training_results)))
	predict_test_results = classifier.predict(test_data)
	results_to_csv(predict_test_results)

problem6(spam_training_data, spam_training_data_labels, spam_test_data, 1)


