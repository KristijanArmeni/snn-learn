# ===== CLASSIFICATION ANALYSIS ===== #

import numpy as np
import sys
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

#Own modules
from util import load, save, Paths

# shortucts to paths
p = Paths()

# 12ax task
stim = load(p.raw + "/12ax-12k.pkl")

# load the actual data
x100 = np.load(p.interim + "/states-100.npy")
x500 = np.load(p.interim + "/states-500.npy")
x1000 = np.load(p.interim + "/states-1000.npy")

# observed responses and stimuli
y_rsp = stim.response  # behavioral responses
y_sym = stim.sequence  # stimulus identity

# responses control
y_rsp0 = np.roll(y_rsp, int((len(y_rsp)/2)))  # shifted responses
y_sym0 = np.roll(y_sym, int((len(y_sym)/2)))  # shifted resposnes


if sys.argv[1] == "validation-curve-response":

    # ===== LOGISTIC REGRESSION: RESPONSES-TRAINING CURVE ===== #
    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, class_weight="balanced", solver="newton-cg", penalty="l2")
    range = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

    trlidx = np.arange(0, 2000)

    for i, x in enumerate([x100, x500, x1000]):

        conditions = ["observed", "permuted"]
        responses = {conditions[0]: y_rsp, conditions[1]: y_rsp0}
        scores = {conditions[0]: None, conditions[1]: None}

        N = x[0, trlidx, :].shape[1]
        x_norm = scaler.fit_transform(x[0, trlidx, :])  # (time_slice, trial, neuron): first dimension stores time window

        for j, condition in enumerate(conditions):

            print("Validation curve for {} condition, N = {} and {} samples".format(condition, N, len(trlidx)))

            y = responses[condition][trlidx]

            # fit logistic regression
            train_sc, val_sc = validation_curve(estimator=logit, X=x_norm, y=y, cv=5, param_name="C",
                                                param_range=range, scoring="balanced_accuracy")

            scores[condition] = [train_sc, val_sc, range]

        save(scores, p.results + "/validation-curve_response-{}.pkl".format(N))


elif sys.argv[1] == "learning-curve-response":

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, class_weight="balanced", C=1.0, solver="newton-cg", penalty="l2")
    sizes = np.array([0.2, 0.4, 0.6, 0.8, 1])

    trlidx = np.arange(0, 2000)

    for i, x in enumerate([x100, x500, x1000]):

        print("Fit nr {}".format(i))

        N = x[0, trlidx, :].shape[1]
        x_norm = scaler.fit_transform(x[0, trlidx, :])  # (time_slice, trial, neuron): first dimension stores time window

        conditions = ["observed", "permuted"]
        responses = {conditions[0]: y_rsp, conditions[1]: y_rsp0}
        scores = {conditions[0]: None, conditions[1]: None}

        for j, condition in enumerate(conditions):

            print("Learning curve for {} condition, N = {} and {} samples".format(condition, N, len(trlidx)))

            y = responses[condition][trlidx]

            # fit logistic regression
            N_sizes, train_sc, val_sc = learning_curve(estimator=logit, X=x_norm, y=y, cv=5,
                                                 train_sizes=sizes, scoring="balanced_accuracy")

            scores[condition] = [train_sc, val_sc, N_sizes]

        save(scores, p.results + "/learning-curve_response-{}.pkl".format(N))

elif sys.argv[1] == "validation-curve-stim":

    # ===== LOGISTIC REGRESSION: STIMULUS IDENTITY, VALIDATION CURVE ===== #

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", max_iter=300, class_weight="balanced",                                  solver="newton-cg", penalty="l2")
    range = np.logspace(-4, 4, 9, base=10)

    for i, x in enumerate([x100, x500, x1000]):

        N = x[0, :, :].shape[1]
        x_norm = scaler.fit_transform(x[0, :, :])  # (time_slice, trial, neuron): first dimension stores time window

        conditions = ["observed", "permuted"]
        responses = {conditions[0]: y_rsp, conditions[1]: y_rsp0}
        scores = {conditions[0]: None, conditions[1]: None}

        for j, condition in enumerate(conditions):

            y = responses[condition]

            print("Validation curve for {} condition and N = {}".format(condition, N))

            # fit logistic regression
            train_sc, val_sc = validation_curve(estimator=logit, X=x_norm, y=y, cv=5, param_name="C",
                                                        param_range=range, scoring="balanced_accuracy")

            scores[condition] = [train_sc, val_sc, range]

        save(scores, p.results + "/validation-curve_symbol-{}.pkl".format(N))

elif sys.argv[1] == "learning-curve-stim":

    # ===== LOGISTIC REGRESSION: STIMULUS IDENTITY, VALIDATION CURVE ===== #

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", C=1.0, max_iter=300, class_weight="balanced",                                  solver="newton-cg", penalty="l2")
    sizes = np.array([0.2, 0.4, 0.6, 0.8, 1])

    for i, x in enumerate([x100, x500, x1000]):

        N = x[0, :, :].shape[1]
        x_norm = scaler.fit_transform(x[0, :, :])  # (time_slice, trial, neuron): first dimension stores time window

        conditions = ["observed", "permuted"]
        responses = {conditions[0]: y_rsp, conditions[1]: y_rsp0}
        scores = {conditions[0]: None, conditions[1]: None}

        for j, condition in enumerate(conditions):

            y = responses[condition]

            print("Learning curve for {} condition and N = {}".format(condition, N))

            # fit logistic regression
            train_sc, val_sc = learning_curve(estimator=logit, X=x_norm, y=y, cv=5,
                                              sizes=sizes, scoring="balanced_accuracy")

            scores[condition] = [train_sc, val_sc, range]

        save(scores, p.results + "/learning-curve_symbol-{}.pkl".format(N))

elif sys.argv[1] == "stimulus-buildup":

    # load data; shape = (tau, time_window, trial, neuron)
    x = np.load(p.interim + "/states_1000-A-0.npy")[0, 1:6, :, :]  # take all time_windows (0 == entire time window)
    responses = {"observed": y_rsp[2000::], "permuted": y_rsp0[2000::]}

    for i, key_y in enumerate(responses):

        y = responses[key_y][2000::]

        scores = []

        # loop over time windows
        for j in range(x.shape[1]):

            print("\n[{:d}] Fitting time window starting at {:d} msec ...".format(j, j*10))

            scaler = StandardScaler()
            logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", C=1.0, max_iter=300,
                                       class_weight="balanced", solver="newton-cg", penalty="l2")

            x_norm = scaler.fit_transform(X=x[i, 2000::, :])  # x_norm.shape = (n_samples, n_neurons)

            accuracy = cross_val_score(estimator=logit, X=x_norm, y=y, cv=5, scoring="balanced_accuracy")
            scores.append(accuracy)

        save(scores, p.results + "/buildup-{}.pkl".format(key_y))


elif sys.argv[1] == "adaptation-curve":

    # load data; shape = (tau, time_window, trial, neuron)
    x_r = np.load(p.interim + "/states-1000_reset_tau.npy")
    x_c = np.load(p.interim + "/states-1000_noreset_tau.npy")

    states = {"reset": x_r, "noreset": x_c}
    responses = {"observed": y_rsp[2000::], "permuted": y_rsp0[2000::]}

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, C=1.0, max_iter=300, class_weight="balanced",
                               solver="newton-cg", penalty="l2")

    # loop over conditions
    for i, key_x in enumerate(states):

        # choose the first time window
        x = states[key_x][:, 0, :, :]  # x.shape = (tau, n_samples, n_neurons)

        # loop over y labels
        for h, key_y in enumerate(responses):

            # choose the samples corresponding to the test set
            y = responses[key_y]  # shape = (n_samples,)
            scores = []

            # loop over tau values
            for j in range(x.shape[0]):

                print("\nRunning states with {}, {} labels and tau iteration {:d}".format(key_x, key_y, j))

                x_norm = scaler.fit_transform(X=x[j, :, :])  # shape=(n_samples, n_features)

                accuracy = cross_val_score(estimator=logit, X=x_norm, y=y, cv=5, scoring="balanced_accuracy")
                scores.append(accuracy)

            save(scores, p.results + "/scores-1000-{}-{}.pkl".format(key_x, key_y))

elif sys.argv[1] == "no-adaptation":

    # load data; shape = (tau, time_window, trial, neuron)
    x_r = np.load(p.interim + "/states-1000_reset_noadapt.npy")

    states = {"reset": x_r}
    responses = {"observed": y_rsp[2000::], "permuted": y_rsp0[2000::]}

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, C=1.0, max_iter=300, class_weight="balanced",
                               solver="newton-cg", penalty="l2")

    # loop over conditions
    for i, key_x in enumerate(states):

        # choose the first time window
        x = states[key_x][:, 0, :, :]  # x.shape = (tau, n_samples, n_neurons)

        # loop over y labels
        for h, key_y in enumerate(responses):

            # choose the samples corresponding to the test set
            y = responses[key_y]  # shape = (n_samples,)
            scores = []

            # loop over tau values
            for j in range(x.shape[0]):

                print("\nRunning states with {}, {} labels and tau iteration {:d}".format(key_x, key_y, j))

                x_norm = scaler.fit_transform(X=x[j, :, :])  # shape=(n_samples, n_features)

                accuracy = cross_val_score(estimator=logit, X=x_norm, y=y, cv=5, scoring="balanced_accuracy")
                scores.append(accuracy)

            save(scores, p.results + "/scores-1000-{}-{}-noadapt.pkl".format(key_x, key_y))