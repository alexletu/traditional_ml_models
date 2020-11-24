import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from save_csv import results_to_csv

for data_name in ["cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)

def permute_dictionaries(training_data, training_labels):
	#takes two dictionaries and permutes both while keeping consistency
	perm = np.random.RandomState(seed=70).permutation(len(training_data))
	return (training_data[perm], training_labels[perm])

cifar10_total_data = io.loadmat("data/%s_data.mat" % "cifar10")

cifar10_training_data = cifar10_total_data["training_data"]
cifar10_training_data_labels = cifar10_total_data["training_labels"]
cifar10_training_data, cifar10_training_data_labels = permute_dictionaries(cifar10_training_data, cifar10_training_data_labels)
cifar10_test_data = cifar10_total_data["test_data"]


cifar10_uncompressed = np.array([np.transpose(pic.reshape(3,32,32), (1,2,0)) for pic in cifar10_training_data])
cifar10_test_uncompressed = np.array([np.transpose(pic.reshape(3,32,32), (1,2,0)) for pic in cifar10_test_data])
#print(cifar10_uncompressed[0].shape)
cifar10_uncompressed_grey = np.array([rgb2gray(pic) for pic in cifar10_uncompressed])
cifar10_test_uncompressed_grey = np.array([rgb2gray(pic) for pic in cifar10_test_uncompressed])
#print(cifar10_uncompressed_grey[0].shape)

hog_features = np.array([hog(image = pic) for pic in cifar10_uncompressed_grey])
hog_test_features =np.array([hog(image = pic) for pic in cifar10_test_uncompressed_grey])

scaler = StandardScaler()
hog_features = scaler.fit_transform(hog_features)
hog_test_features = scaler.transform(hog_test_features)

print("cifar10_training_data", hog_features.shape)
print("cifar10_training_data_labels", cifar10_training_data_labels.shape)
print("cifar10_test_data", hog_test_features.shape)

def problem6(training_data, training_data_labels, test_data, linear, C_Value = 0):

	classifier = svm.LinearSVC(dual = False, random_state = 10, verbose = 1, max_iter = 1000000)

	if(not linear):
		classifier = svm.SVC(kernel = "linear", random_state = 10, verbose = 0)

	classifier.fit(training_data, np.ravel(training_data_labels))

	predict_training_results = classifier.predict(training_data)
	print(accuracy_score(np.ravel(training_data_labels), np.ravel(predict_training_results)))
	predict_test_results = classifier.predict(test_data)
	results_to_csv(predict_test_results)

problem6(hog_features, cifar10_training_data_labels, hog_test_features, True, 0)

