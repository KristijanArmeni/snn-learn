
from matplotlib import pyplot as plt
import numpy as np


def plot_trial(model=None, trial=None, variable=None):

    plt.figure()
    plt.imshow(model.recording[variable][trial-1, :, :], aspect="auto")
    plt.xlabel("Time (msec)")
    plt.ylabel("neuron")
    plt.title("Trial {}".format(trial))
    if variable == "V":
        c = plt.colorbar()
        c.set_label("Voltage (mV)")
    plt.show()


def plot_time(model=None, trial=None, neurons=None, variable=None):

    plt.figure()
    plt.plot(model.recording[variable][trial-1, :, :].T)
    plt.xlabel("Time (msec)")
    plt.ylabel("membrane potential (mV)")
    plt.title("Trial {}".format(trial))
    plt.show()


# convenience function for plotting classifier output
def plot_scores(scores, title=None, xlabel=None, ylabel=None):

    fig, ax = plt.subplots(1, len(scores))
    fig.suptitle(title, size=18)

    for i, key in enumerate(scores):

        data = scores[key]
        X = data[2]

        # compute mean accuracy scores
        train_mean = np.mean(data[0], 1)
        test_mean = np.mean(data[1], 1)
        train_std = np.std(data[0], 1)
        test_std = np.std(data[1], 1)

        ax[i].fill_between(X, train_mean - train_std, train_mean + train_std,
                           alpha=0.1)
        ax[i].fill_between(X, test_mean - test_std, test_mean + test_std,
                           alpha=0.1)

        ax[i].semilogx(X, np.mean(data[0], 1), '--o', label='training score')
        ax[i].semilogx(X, np.mean(data[1], 1), '--o', label='test score')

        ax[i].set_ylim(0, 1.05)
        ax[i].set_xlim(X[0], X[-1])
        ax[i].set_xlabel(xlabel)
        if i == 0:
            ax[i].set_ylabel(ylabel)
        ax[i].set_title(key, size=14)
        ax[i].legend(loc='best')
