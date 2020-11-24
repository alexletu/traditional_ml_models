import numpy as np
import gc
from scipy import io
from scipy.stats import multivariate_normal
import cv2
from save_csv import results_to_csv
SZ = 28

winSize = (28, 28)
blockSize = (12, 12)
blockStride = (4, 4)
cellSize = (12, 12)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, signedGradients)

def permute_dictionaries(data, labels, rand=25):
    perm = np.random.RandomState(seed=rand).permutation(training_data.shape[0])
    return data[perm], labels[perm]

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def deskew_all(data):
    new_data = []
    for i in range(data.shape[0]):
        row_img = data[i]
        img = row_img.reshape((28, 28))
        new_data.append(np.array((hog.compute(deskew(img)))).flatten())
    return np.array(new_data)

gc.enable()

mnist_data = io.loadmat("mnist-data/mnist_data.mat")
print("Loaded mnist data.")
training_data = mnist_data["training_data"]
training_labels = mnist_data["training_labels"]

training_data, training_labels = permute_dictionaries(training_data, training_labels,1000)
training_data = deskew_all(training_data)

training_data, validation_data = training_data[:50000], training_data[50000:]
training_labels, validation_labels = training_labels[:50000], training_labels[50000:]

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n, features = training_data.shape

print("\nTraining data: ", training_data.shape)
print("Training data labels: ", training_labels.shape)
print("Validation data: ", validation_data.shape)
print("Validation labels: ", validation_labels.shape)

def empirical_mean(partitioned_data):
    return {k : np.sum(partitioned_data[k], 0, keepdims=True).transpose() / len(partitioned_data[k]) for k in classes}

def empirical_cov(partitioned_data):
    return {k : np.cov(partitioned_data[k].T, bias=True) for k in classes}

def calc_priors(partitioned_data, total):
    return {k: partitioned_data[k].shape[0] / total for k in classes}

def partition_data(data, labels):
    partitioned = {k: [] for k in classes}
    for sample_num in range(data.shape[0]):
        k = labels[sample_num][0]
        sample_features = data[sample_num]
        partitioned[k].append(sample_features)
    for k in classes:
        partitioned[k] = np.asarray(partitioned[k])
    return partitioned

def error_rate(prediction, actual):
    assert len(prediction) == len(actual)
    return np.count_nonzero(prediction - actual) / prediction.shape[0]

def classify(distributions, samples, priors):
    all_predictions = {}
    for key in samples.keys():
        predictions = []
        for sample in samples[key]:
            ll = {k: 0 for k in classes}
            for k in classes:
                sample = np.array(sample)
                ll[k] = distributions[k].logpdf(sample) + np.log(priors[k])
            predictions.append(max(ll, key=lambda key: ll[key]))
        all_predictions[key] = predictions
    return all_predictions

def pool_cov(covariances, priors):
    cov = np.zeros(covariances[0].shape)
    for k in classes:
        cov += priors[k] * covariances[k]
    return cov

def LDA(means, covariances, priors, inputs, c=0.0):
    pooled_cov = pool_cov(covariances, priors)
    pooled_cov += np.eye(features) * c * np.trace(pooled_cov)
    distributions = {k: multivariate_normal(means[k].flatten(), pooled_cov, allow_singular=True) for k in classes}
    return classify(distributions, inputs, priors)
def QDA(means, covariances, priors, inputs, c=0.0):
    temp_covariances, distributions = {}, {}
    for k in classes:
        temp_covariances[k] = np.eye(features) * c * np.trace(covariances[k]) + covariances[k]
        distributions[k] = multivariate_normal(means[k].flatten(), temp_covariances[k], allow_singular=True)
    return classify(distributions, inputs, priors)


"""------------------------------------------------------------------------------------------------------------------"""
"""------------------------------------------------------------------------------------------------------------------"""
"""------------------------------------------------------------------------------------------------------------------"""

def test_QDA(training_data, training_labels, validation_data, validation_labels, c=0.0):
    partitioned_training_data = partition_data(training_data, training_labels)
    means = empirical_mean(partitioned_training_data)
    covariances = empirical_cov(partitioned_training_data)
    priors = calc_priors(partitioned_training_data, training_data.shape[0])
    samples = {'validation' : validation_data}
    predictions = QDA(means, covariances, priors, samples, c)
    return error_rate(np.array([predictions['validation']]).T, validation_labels)

def test_LDA(training_data, training_labels, validation_data, validation_labels, c=0.0):
    partitioned_training_data = partition_data(training_data, training_labels)
    means = empirical_mean(partitioned_training_data)
    covariances = empirical_cov(partitioned_training_data)
    priors = calc_priors(partitioned_training_data, training_data.shape[0])
    samples = {'validation' : validation_data}
    predictions = LDA(means, covariances, priors, samples, c)
    return error_rate(np.array([predictions['validation']]).T, validation_labels)

#print(test_QDA(training_data, training_labels, validation_data, validation_labels, .0001))

def kaggle(c):
    data = deskew_all(mnist_data["training_data"])
    labels = mnist_data["training_labels"]
    test_data = deskew_all(mnist_data["test_data"])
    partitioned_data = partition_data(data, labels)

    means = empirical_mean(partitioned_data)
    partitioned_covariances = empirical_cov(partitioned_data)
    priors = calc_priors(partitioned_data, len(data))
    samples = {'training' : data, 'test' : test_data}

    predictions = QDA(means, partitioned_covariances, priors, samples, c)
    train_predictions = predictions['training']
    test_predictions = predictions['test']
    print(error_rate(np.array([train_predictions]).T, labels))
    results_to_csv(np.array(test_predictions))
    return
#kaggle(.0001)

def opt_c_value_QDA(training_data, training_labels, validation_data, validation_labels, c_values):
    results = {}
    for c in c_values:
        results[c] = test_QDA(training_data,training_labels,validation_data,validation_labels,c)
        print("Error rate ", results[c], " achieved with c value: ", c)
    best_c = min(results, key=lambda key: results[key])
    print("Optimal c_value was ", best_c, " with error: ", results[best_c])
    return best_c, results[best_c]

def opt_c_value_LDA(training_data, training_labels, validation_data, validation_labels, c_values):
    results = {}
    for c in c_values:
        results[c] = test_LDA(training_data,training_labels,validation_data,validation_labels,c)
        print("Error rate ", results[c], " achieved with c value: ", c)
    best_c = min(results, key=lambda key: results[key])
    print("Optimal c_value was ", best_c, " with error: ", results[best_c])
    return best_c

def gen_c_values(low_exp, high_exp):
    return [10**i for i in range(low_exp, high_exp + 1)]

print(opt_c_value_QDA(training_data, training_labels, validation_data, validation_labels, np.arange(.00047, .008, .0000001)))
