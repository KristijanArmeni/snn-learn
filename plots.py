
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

# convenience function for plotting classifier output
def plot_scores(scores, title=None):

    fig, ax = plt.subplots(1, 3)
    fig.suptitle(title, size=18)

    for i, data in enumerate(scores):

        N = data[3]

        # compute mean accuracy scores
        train_mean = np.mean(data[1], 1)
        test_mean = np.mean(data[2], 1)
        train_std = np.std(data[1], 1)
        test_std = np.std(data[2], 1)

        ax[i].fill_between(N, train_mean - train_std, train_mean + train_std,
                           alpha=0.1)
        ax[i].fill_between(N, test_mean - test_std, test_mean + test_std,
                           alpha=0.1)

        ax[i].plot(N, np.mean(data[1], 1), '--o', label='training score')
        ax[i].plot(N, np.mean(data[2], 1), '--o', label='test score')

        ax[i].set_ylim(0, 1.05)
        ax[i].set_xlim(N[0], N[-1])
        ax[i].set_xlabel('training set size (num. samples)')
        if i == 0:
            ax[i].set_ylabel('average classification accuracy (prop. correct)')
        ax[i].set_title(data[0], size=14)
        ax[i].legend(loc='best')