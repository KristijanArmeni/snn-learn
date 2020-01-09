# ===== CLASSIFICATION ANALYSIS ===== #

import numpy as np
import sys
import gc
from matplotlib import pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.linear_model import LogisticRegression

# Own modules
from util import load, save, Paths

# shortucts to paths
p = Paths()

# 12ax task
stim = load(p.raw + "/12ax-12k.pkl")

# observed responses and stimuli
y_rsp = stim.response  # behavioral responses
y_sym = stim.sequence  # stimulus identity

# downsample the majority class (== False)
selTrue = np.where(y_rsp == True)[0]
selFalse = np.random.RandomState(123).choice(np.where(y_rsp==False)[0], size=len(selTrue), replace=False)
sel = np.concatenate((selTrue, selFalse))

# find target ids
t1 = np.where(y_sym == 'X')[0]
t2 = np.where(y_sym == 'Y')[0]
tgt = np.concatenate((t1, t2))  # only targets, Y and X

# now find a conjuct
tgtselid = np.isin(sel, tgt)  # check whether downsampled stims are targets
tgtsel = sel[tgtselid]

# responses control
y_rsp0 = np.roll(y_rsp, int((len(y_rsp)/2)))  # shifted responses
y_sym0 = np.roll(y_sym, int((len(y_sym)/2)))  # shifted responses

# define dependency length

seg, _, _, pseg = stim.segment()
lev = np.concatenate([np.arange(0, len(i)) for i in seg])
lev[np.isin(lev, np.array([1, 2]))] = 1
lev[np.isin(lev, np.array([3, 4]))] = 2
lev[np.isin(lev, np.array([5, 6]))] = 3
lev[np.isin(lev, np.array([7, 8]))] = 4
levels = np.asarray(lev)


# Balance datasets
maxsamples = np.sum(levels[2000::] == 4)  # the max number of samples should correspond to the number of level 4 positions in the train set
levIds = [np.sort(np.random.RandomState(level_seed[1]).choice(a=np.where(levels == level_seed[0])[0][np.where(levels == level_seed[0])[0]>2000],
                                                              size=maxsamples))
                                                              for level_seed in [(1, 12), (2, 30), (3, 50), (4, 70)]]

# quit test that it works
assert np.sum(levels[levIds[0]] == 1) == maxsamples
assert np.sum(levels[levIds[1]] == 2) == maxsamples
assert np.sum(levels[levIds[2]] == 3) == maxsamples
assert np.sum(levels[levIds[3]] == 4) == maxsamples

if sys.argv[1] == "validation-curve-response":

    # ===== LOGISTIC REGRESSION: RESPONSES-TRAINING CURVE ===== #
    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, class_weight=None, solver="newton-cg", penalty="l2")
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
    logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", max_iter=300, class_weight="balanced",
                               solver="newton-cg", penalty="l2")
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

elif sys.argv[1] == "stimulus-decoding-time":

    # Subject loop
    for h in np.arange(0, 10):

        print("\n Running subject s{:02d} ...".format(h+1))

        # load data; shape = (tau, time_window, trial, neuron)
        data = np.load(p.interim + "/states_1000-A-s{:02d}.npy".format(h + 1))[4, 1::, :, :]   # 4 --> tau == 400 msec; 1-end --> time steps

        # create dataset for new and old stimulus labels
        x = data[:, 1::, :]              # from the second sample onwards
        y_sym_new = y_sym[2001::]        # from the second sample onwards
        y_sym_new0 = y_sym0[2001::]      # from the second sample onwards
        y_sym_old = y_sym[2000:-1]       # from the first sample onwards
        y_sym_old0 = y_sym0[2000:-1]

        responses = {"new-observed": y_sym_new,
                     #"new-permuted": y_sym_new0,
                     "old-observed": y_sym_old}
                     #"old-permuted": y_sym_old0}

        # response loop
        for i, key_y in enumerate(responses):

            print("Fitting model with {} responses ...".format(key_y))
            y = responses[key_y]
            scores = []

            # loop over time windows x.shape = (time, trials, neurons)
            for j in range(x.shape[0]):

                print("[{:d}] Fitting time window starting at {:d} msec ...".format(j, j*10))

                scaler = StandardScaler()
                logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", C=1.0, max_iter=300,
                                           class_weight="balanced", solver="newton-cg", penalty="l2")

                x_norm = scaler.fit_transform(X=x[j, :, :])  # x_norm.shape = (n_samples, n_neurons)

                accuracy = cross_val_score(estimator=logit, X=x_norm, y=y, cv=5, scoring="balanced_accuracy")
                scores.append(accuracy)

            save(scores, p.results + "/stimulus-decoding_{}_s{:02d}.pkl".format(key_y, h+1))

        # delete names, free memory
        del data, x
        gc.collect()

elif sys.argv[1] == "stimulus-decoding-adaptation":

    pass

elif sys.argv[1] == "adaptation-curve":

    N = 1000
    targetsOnly = False
    balanceDs = False
    level = 1

    # select targets only
    print("TargetsOnly == {}".format(targetsOnly))
    print("Balance == {}".format(balanceDs))
    print("Level == {}".format(level))

    # subselect features if necessary
    selection = None
    if (level is None) and targetsOnly and not balanceDs:
        selection = tgt[tgt > 2000] - 2000
    elif (level is None) and balanceDs and not targetsOnly:
        selection = sel[sel > 2000] - 2000
    elif (level is None) and balanceDs and targetsOnly:
        selection = tgtsel[tgtsel > 2000] - 2000
    elif level is not None and not balanceDs:
        selection = np.isin(lev[2000::], level)
    elif (level is not None) and targetsOnly and not balanceDs:
        selection = np.isin(lev[tgt][2000::])
    elif (level is not None) and balanceDs:
        selection = levIds[level-1]

    # load data; shape = (tau, time_window, trial, neuron)
    subjects = np.arange(0, 10)

    scaler = StandardScaler()
    logit = LogisticRegression(fit_intercept=True, C=1.0, max_iter=300, class_weight="balanced",
                               solver="newton-cg", penalty="l2")

    # loop over subjects
    for k in subjects:

        print("Running subject s{:02d} (N = {})".format(k+1, N))
        fname = os.path.join(p.interim, "newtime", "states_{}-A-s{:02d}.npy".format(N, subjects[k]+1))

        if selection is None:
            print("Loading {} ...".format(fname))
            x_r = np.load(fname)
            y_rsp = stim.response[2000::]
        else:
            print("loading {} and applying selection ...".format(fname))
            x_r = np.load(fname)[:, :, selection, :]
            y_rsp = (stim.response[2000::])[selection]

        # make a quick test that balancing works
        if balanceDs:
            assert np.sum(levels[selection + 2000] == level) == maxsamples

        states = {"A": x_r}
        responses = {"observed": y_rsp}

        # define scoring functions
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'kappa': make_scorer(cohen_kappa_score),
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall'
        }

        # loop over conditions
        for i, key_x in enumerate(states):

            # choose the first time window (== average)
            x = states[key_x][:, 0, :, :]  # x.shape = (tau, time_windows, n_samples, n_neurons)

            # loop over y labels
            for h, key_y in enumerate(responses):

                # choose the samples corresponding to the test set
                y = responses[key_y]  # shape = (n_samples,)
                scores = []

                # loop over tau values
                for j in range(x.shape[0]):

                    print("\nClassifying states with {}, {} labels and tau iteration {:d}".format(key_x, key_y, j))

                    x_norm = scaler.fit_transform(X=x[j, :, :])  # shape=(n_samples, n_features)

                    accuracy = cross_validate(estimator=logit, X=x_norm, y=y,
                                              cv=5, scoring=scoring,
                                              return_train_score=True, return_estimator=True)

                    scores.append(accuracy)

                # construct the name for saving
                savename = None
                suffix = None

                if targetsOnly and not balanceDs:
                    suffix = 'tgt'
                elif balanceDs and not targetsOnly:
                    suffix = 'bal'
                elif balanceDs and targetsOnly:
                    suffix = 'tgtbal'
                elif not balanceDs and not targetsOnly and level is None:
                    suffix = None
                elif not balanceDs and not targetsOnly and level is not None:
                    suffix = "{}".format(level)
                elif balanceDs and not targetsOnly and level is not None:
                    suffix = "{}_bal".format(level)

                if suffix is not None:
                    savename = "scores_{}-{}-{}-s{:02d}_{}.pkl".format(N, key_x, key_y, k + 1, suffix)
                else:
                    savename = "scores_{}-{}-{}-s{:02d}.pkl".format(N, key_x, key_y, k + 1)

                print("Saving {}".format(os.path.join(p.results, 'newtime', savename)))
                save(scores, os.path.join(p.results, 'newtime', savename))

            del x
            gc.collect()

        # clear variable from memory
        del x_r, states, responses
        gc.collect()

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

                accuracy = cross_validate(estimator=logit, X=x_norm, y=y,
                                          cv=5, scoring=('balanced_accuracy', 'accuracy', 'precision', 'recall'),
                                          return_train_score=True, return_estimator=True)

                scores.append(accuracy)

            save(scores, p.results + "/scores-1000-{}-{}-noadapt.pkl".format(key_x, key_y))

elif sys.argv[1] == 'baseline':

    logit = LogisticRegression(fit_intercept=True, C=1.0, max_iter=300, class_weight='balanced',
                               solver="newton-cg", penalty="l2")

    scoring = ('balanced_accuracy', 'accuracy', 'precision', 'recall')

    print("Classifying resposnes based on stimuli")
    scores = cross_validate(estimator=logit, X=stim.encoding[:, 2000::].T, y=stim.response[2000::],
                            cv=5, scoring=scoring, return_estimator=True, return_train_score=True)

    save(scores, p.results + "/scores-baseline.pkl")

