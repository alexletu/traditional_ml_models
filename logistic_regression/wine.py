import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from save_csv import results_to_csv
from sklearn.preprocessing import StandardScaler
import time


def permute_dictionaries(data, labels, rand=0):
    perm = np.random.RandomState(seed=rand).permutation(data.shape[0])
    return data[perm], labels[perm]

temp = io.loadmat("data.mat")
data = temp["X"]
labels = temp["y"]
data, labels = permute_dictionaries(data, labels)
print("Loaded wine data.")

train_data, valid_data = data[:5500], data[5500:]
train_labels, valid_labels = labels[:5500], labels[5500:]
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valid_data = scaler.transform(valid_data)
train_data = np.hstack((train_data, np.ones((train_data.shape[0], 1))))
valid_data = np.hstack((valid_data, np.ones((valid_data.shape[0], 1))))

def gen_c_values(low_exp, high_exp):
    return [10**i for i in range(low_exp, high_exp + 1)]

def error_rate(prediction, actual):
    return np.count_nonzero(prediction - actual) / prediction.shape[0]

def print_info(title, time, alpha, c, epsilon, n_samples, n_test, n_iterations, final_loss, train_error, test_error, decay = None):
    print("********** - %s - **********" % title)
    print("     Time elapsed:            %f" % time)
    print("     Learning rate:           %f" % alpha)
    print("     Regularization value:    %f" % c)
    if decay is not None:
        print("     Decay Rate:           %.10f" % decay)
    print("     Epsilon value:           %f" % epsilon)
    print("     Training Samples:        %d" % n_samples)
    print("     Test samples:            %d" % n_test)
    print("     Total iterations:        %d" % n_iterations)
    print("     Final Loss:              %f" % final_loss)
    print("     Final train error        %f" % train_error)
    print("     Final error test error   %f" % test_error)
    return

def plot_data(iterations, losses, clr, title):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.set_dpi(100)
    plt.subplot(1, 1, 1)
    plt.plot(iterations, losses, label="Loss", color=clr, marker='.', linestyle='dashed',linewidth=1, markersize=1)
    plt.legend()
    plt.title(title + " vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


"""------------------------------------------------------------------------------------------------------------------"""


def sigmoid(gamma):
    gamma[gamma <= -709] = -700
    gamma[gamma >= 709] = 700
    return 1/(1 + np.e**(-gamma))

def log_loss(z, y):
    return -y * np.log(z + 10**-300) - (1 - y) * np.log(1 - z + 10**-300)

def obj(act, pred, w, c):
    return np.mean(log_loss(pred, act)) + (c / 2) * np.linalg.norm(w)

def calc_gradient(X, y, predictions, w, c):
    n = X.shape[0]
    dz = predictions - y
    dw = (1 / n) * np.matmul(X.T, dz) + c * w
    return dw

def classify(data, w):
    probabilities = np.where(sigmoid(np.matmul(data, w)) > .5, 1, 0)
    return probabilities

def train_batch(X, x_labels, alpha, c, epsilon):
    w = np.zeros((13, 1))
    iteration = 0
    loss = np.inf
    iterations = []
    losses = []
    w_prev = np.ones((13,1))
    while not np.allclose(w_prev, w) and loss >= epsilon:
        x_predictions = sigmoid(np.matmul(X, w))
        dw = calc_gradient(X, x_labels, x_predictions, w, c)
        w_prev = w
        w = w - alpha * dw
        loss = obj(x_labels, x_predictions, w, c)
        iteration += 1
        if iteration % 10 == 0:
            iterations.append(iteration)
            losses.append(loss)
            print(loss)
    iterations.append(iteration)
    losses.append(loss)
    return w, iterations, losses

def train_sgd(X, x_labels, alpha, c, epsilon, decay=0):
    w = np.zeros((13, 1))
    iteration = 0
    loss = np.inf
    iterations = []
    losses = []
    i = 0
    w_prev = np.ones((13, 1))
    while not np.allclose(w_prev, w) and loss >= epsilon:
        #out of all x choose an index i, convert it to matrix
        x_i = np.array([X[i]])
        x_i_prediction = sigmoid(np.matmul(x_i, w))
        #convert the label to the single index i, convert to matrix
        l_i = x_labels[i]

        dw = calc_gradient(x_i, l_i, x_i_prediction, w, c)
        w = w - alpha * dw
        all_predictions = sigmoid(np.matmul(X, w))
        loss = obj(x_labels, all_predictions, w, c)
        alpha *= 1 / (1 + decay * iteration)
        if iteration % 10 == 0:
            iterations.append(iteration)
            losses.append(loss)
            print(loss)
        iteration += 1
        i += 1
        if i == X.shape[0]:
            X, x_labels = permute_dictionaries(X, x_labels, rand=np.random.randint(0, 100))
            i = 0

    iterations.append(iteration)
    losses.append(loss)
    return w, iterations, losses

def part_a(alpha, c, epsilon):
    start = time.time()
    w, iterations, losses = train_batch(train_data, train_labels, alpha, c, epsilon)
    time_elapsed = time.time() - start
    error = error_rate(classify(valid_data, w), valid_labels)
    train_error = error_rate(classify(train_data, w), train_labels)
    print_info("Summary Batch Gradient Decent", time_elapsed, alpha, c, epsilon, train_data.shape[0], valid_data.shape[0],  iterations[len(iterations) - 1],losses[len(losses) - 1] ,train_error, error)
    plot_data(iterations, losses, 'red', "Batch Gradient Decent")

def part_b(alpha, c, epsilon):
    start = time.time()
    w, iterations, losses = train_sgd(train_data, train_labels, alpha, c, epsilon)
    time_elapsed = time.time() - start
    error = error_rate(classify(valid_data, w), valid_labels)
    train_error = error_rate(classify(train_data, w), train_labels)
    print_info("Summary Stochastic Gradient Decent", time_elapsed, alpha, c, epsilon, train_data.shape[0], valid_data.shape[0],  iterations[len(iterations) - 1], losses[len(losses) - 1],train_error, error)
    plot_data(iterations, losses, 'blue', 'Stochastic Gradient Descent Loss')

def part_c(alpha, c, epsilon, decay):
    start = time.time()
    w, iterations, losses = train_sgd(train_data, train_labels, alpha, c, epsilon, decay)
    time_elapsed = time.time() - start
    error = error_rate(classify(valid_data, w), valid_labels)
    train_error = error_rate(classify(train_data, w), train_labels)
    print_info("SGD w/ decreasing learning rate", time_elapsed, alpha, c, epsilon, train_data.shape[0], valid_data.shape[0],  iterations[len(iterations) - 1], losses[len(losses) - 1], train_error, error, decay)
    plot_data(iterations, losses, 'green', 'SGD w/ Decreasing Dearning Date Loss')

def kaggle(alpha, c, epsilon):
    X = temp["X"]
    test = temp["X_test"]
    x_labels = temp["y"]
    X, x_labels = permute_dictionaries(X, x_labels)
    X = scaler.transform(X)
    test = scaler.transform(test)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    test = np.hstack((test, np.ones((test.shape[0], 1))))
    w = np.zeros((13, 1))

    epoch = 0
    loss = np.inf
    while loss >= epsilon:
        x_predictions = sigmoid(np.matmul(X, w))

        dw = calc_gradient(X, x_labels, x_predictions, w, c)
        w = w - alpha * dw
        loss = obj(x_labels, x_predictions, w, c)
        epoch += 1
        if epoch % 1000 == 0:
            print(loss)
    test_predictions = classify(test, w)
    print(test_predictions)
    results_to_csv(test_predictions.flatten())

part_a(.02, 0, .0356)
part_b(.02, 0, .0356)
part_c(.02, 0, .0356, .0000000001)

#kaggle(1, 0, .03465751782558396)