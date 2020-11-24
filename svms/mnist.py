import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from sklearn.metrics import accuracy_score
from sklearn.kernel_approximation import Nystroem
from save_csv import results_to_csv

for data_name in ["mnist"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)

def permute_dictionaries(training_data, training_labels):
	#takes two dictionaries and permutes both while keeping consistency
	perm = np.random.RandomState(seed=70).permutation(len(training_data))
	return (training_data[perm], training_labels[perm])

total_data = io.loadmat("data/%s_data.mat" % "mnist")

index = 60000
total_training_data = total_data["training_data"] / float(255)
total_training_data_labels = total_data["training_labels"]
total_training_data, total_training_data_labels = permute_dictionaries(total_training_data, total_training_data_labels)
test_data = total_data["test_data"] / float(255)

feature_map_nystroem = Nystroem(gamma = .05, n_components = 25000)
features_training = feature_map_nystroem.fit_transform(total_training_data)
features_test = feature_map_nystroem.transform(test_data)

print("mnist_training_data", features_training.shape)
print("mnist_training_data_labels", total_training_data_labels.shape)
print("mnist_test_data", features_test.shape)

def problem5(training_data, training_data_labels, test_data, C_value):	
	classifier = svm.LinearSVC(dual = False, random_state = 10, C = C_value)

	classifier.fit(training_data, np.ravel(training_data_labels))

	predict_training_results = classifier.predict(training_data)
	print(accuracy_score(np.ravel(training_data_labels), np.ravel(predict_training_results)))
	predict_test_results = classifier.predict(test_data)
	results_to_csv(predict_test_results)


problem5(features_training, total_training_data_labels, features_test, 5)