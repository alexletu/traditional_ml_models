import numpy as np
import gc
from scipy import io
from scipy.stats import multivariate_normal
from save_csv import results_to_csv

def sparse_to_np(sparse):
    temp = []
    for samp in range(sparse.shape[0]):
        row = sparse[samp].toarray()[0]
        temp.append(row)
    return np.asarray(temp)
def permute_dictionaries(data, labels, rand=25):
    perm = np.random.RandomState(seed=rand).permutation(training_data.shape[0])
    return data[perm], labels[perm]


gc.enable()

spam_data = io.loadmat("spam-data/spam_data.mat")
print("Loaded spam data.")
training_data = sparse_to_np(spam_data["training_data"])
training_labels = spam_data["training_labels"]

training_data, training_labels = permute_dictionaries(training_data, training_labels)
training_data, validation_data = training_data[:4138], training_data[4138:]
training_labels, validation_labels = training_labels[:4138], training_labels[4138:]

classes = [0, 1]
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

#print(test_QDA(training_data, training_labels, validation_data, validation_labels, .00064))
#print(test_LDA(training_data, training_labels, validation_data, validation_labels))

def kaggle(c):
    data = sparse_to_np(spam_data["training_data"])
    labels = spam_data["training_labels"]
    test_data = sparse_to_np(spam_data["test_data"])
    partitioned_data = partition_data(data, labels)

    means = empirical_mean(partitioned_data)
    partitioned_covariances = empirical_cov(partitioned_data)
    priors = calc_priors(partitioned_data, len(data))
    samples = {'training' : data}

    predictions = QDA(means, partitioned_covariances, priors, samples, c)
    train_predictions = predictions['training']
    #test_predictions = predictions['test']
    print(error_rate(np.array([train_predictions]).T, labels))
    #results_to_csv(np.array(test_predictions))
    #return

#print(kaggle(.00064))
#0.0004833252779120348 train error with 5000 @ .00064 no prior weighting (~95% test accuracy)

def opt_c_value(training_data, training_labels, validation_data, validation_labels, c_values):
    results = {}
    for c in c_values:
        results[c] = k_fold(training_data, training_labels, 5, c)
        print("Error rate ", results[c], " achieved with c value: ", c)
    best_c = min(results, key=lambda key: results[key])
    print("Optimal c_value was ", best_c, " with error: ", results[best_c])
    return best_c

def k_fold(data, labels, k, c):
    data, labels = permute_dictionaries(data, labels, np.random.randint(0,10000))
    data_partitions = np.array_split(data, k)
    label_partitions = np.array_split(labels, k)
    errors = []
    for k in range(k):
        validation_data = data_partitions[0]
        validation_labels = label_partitions[0]
        training_data = np.concatenate(data_partitions[1:])
        training_labels = np.concatenate(label_partitions[1:])

        error = test_QDA(training_data, training_labels, validation_data, validation_labels, c)

        data_partitions = np.roll(data_partitions, 1)
        label_partitions = np.roll(label_partitions, 1)
        errors.append(error)
    return sum(errors) / k

#opt_c_value(training_data, training_labels, validation_data, validation_labels, np.arange(.0006, .0007, .0001))


