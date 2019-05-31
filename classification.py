# ===== CLASSIFICATION ANALYSIS ===== #

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

#Own modules
from util import load, save
from plots import plot_scores

# load the actual data
x100 = load("/project/3011085.04/snn/data/interim/train_x-100.pkl")
x500 = load("/project/3011085.04/snn/data/interim/train_x-500.pkl")
x1000 = load("/project/3011085.04/snn/data/interim/train_x-1000.pkl")

# load the test data
test_x100 = load("/project/3011085.04/snn/data/interim/test_x-100.pkl")
test_x500 = load("/project/3011085.04/snn/data/interim/test_x-500.pkl")
test_x1000 = load("/project/3011085.04/snn/data/interim/test_x-1000.pkl")

y = load("/project/3011085.04/snn/data/interim/train_ds.pkl")
y_test = load("/project/3011085.04/snn/data/interim/test_ds.pkl")

# observed responses and stimuli
y_rsp = y.response  # behavioral responses
y_sym = y.sequence  # stimulus identity
y_test_rsp = y_test.response
y_test_sym = y_test.sequence

# responses control
y_rsp0_1 = np.random.RandomState(55).permutation(y_rsp)   # permuted responses
y_rsp0_2 = np.roll(y_rsp, int((len(y_rsp)/2)))            # shifted resposnes

y_test_rsp0_1 = np.random.RandomState(55).permutation(y_test_rsp)   # permuted responses
y_test_rsp0_2 = np.roll(y_test_rsp, int((len(y_test_rsp)/2)))            # shifted resposnes

# stimulus identity control
y_sym0_1 = np.random.RandomState(55).permutation(y_sym)   # permuted responses
y_sym0_2 = np.roll(y_sym, int((len(y_sym)/2)))            # shifted resposnes

y_test_sym0_1 = np.random.RandomState(55).permutation(y_test_sym)   # permuted responses
y_test_sym0_2 = np.roll(y_test_sym, int((len(y_test_sym)/2)))        # shifted resposnes

save(y_rsp0_1, "/project/3011085.04/snn/data/interim/train_y_rsp_perm.pkl")
save(y_rsp0_2, "/project/3011085.04/snn/data/interim/train_y_rsp_shift.pkl")
save(y_sym0_1, "/project/3011085.04/snn/data/interim/train_y_sym_perm.pkl")
save(y_sym0_2, "/project/3011085.04/snn/data/interim/train_y_sym_shift.pkl")


# ===== LOGISTIC REGRESSION: RESPONSES-TRAINING CURVE ===== #

scaler = StandardScaler()

logit = LogisticRegression(fit_intercept=True, class_weight="balanced", solver="newton-cg", penalty="l2")

reg = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

scores = []

for i, x in enumerate([test_x100, test_x500, test_x1000]):

    print("Fit nr {}".format(i))

    N = x[0, :, :].shape[1]

    x_norm = scaler.fit_transform(x[0, :, :])  # (time_slice, trial, neuron): first dimension stores time window

    # fit logistic regression
    train_sc_obs, val_sc_obs = validation_curve(estimator=logit, X=x_norm, y=y_rsp, cv=2, param_name="C",
                                                param_range=reg, scoring="balanced_accuracy")
    train_sc_rnd, val_sc_rnd = validation_curve(estimator=logit, X=x_norm, y=y_rsp0_1, cv=2, param_name="C",
                                                param_range=reg, scoring="balanced_accuracy")
    train_sc_shf, val_sc_shf = validation_curve(estimator=logit, X=x_norm, y=y_rsp0_2, cv=2, param_name="C", param_range=reg,
                                                scoring="balanced_accuracy")

    scores.append([["observed", train_sc_obs, val_sc_obs, reg],
                ["permuted", train_sc_rnd, val_sc_rnd, reg],
                ["shifted", train_sc_shf, val_sc_shf, reg]])

    save(scores, "/project/3011085.04/snn/data/results/validation-curve_LR-{}.pkl".format(N))

plot_scores(exp[0], title="Validation curve: Logistic regression (100 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")

plot_scores(exp[1], title="Validation curve: Logistic regression (500 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")

plot_scores(exp[2], title="Validation curve: Logistic regression (1000 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")


scaler = StandardScaler()

logit = LogisticRegression(fit_intercept=True, class_weight="balanced", solver="newton-cg", penalty="l2")

sizes = np.array([0.2, 0.4, 0.6, 0.8, 1])

exp = []

for i, x in enumerate([test_x100, test_x500, test_x1000]):

    print("Fit nr {}".format(i))

    N = x[0, :, :].shape[1]

    x_norm = scaler.fit_transform(x[0, :, :])  # (time_slice, trial, neuron): first dimension stores time window

    # fit logistic regression
    N, train_sc_obs, val_sc_obs = learning_curve(estimator=logit, X=x_norm, y=y_test_rsp, cv=5, train_sizes=sizes,
                                              scoring="balanced_accuracy")
    N, train_sc_rnd, val_sc_rnd = learning_curve(estimator=logit, X=x_norm, y=y_test_rsp0_1, cv=5, train_sizes=sizes,
                                              scoring="balanced_accuracy")
    N, train_sc_shf, val_sc_shf = learning_curve(estimator=logit, X=x_norm, y=y_test_rsp0_2, cv=5, train_sizes=sizes,
                                              scoring="balanced_accuracy")

    exp.append([["observed", train_sc_obs, val_sc_obs, N],
                ["permuted", train_sc_rnd, val_sc_rnd, N],
                ["shifted", train_sc_shf, val_sc_shf, N]])

    save(exp, "/project/3011085.04/snn/data/results/learning-curve2_LR-{}.pkl".format(N))

plot_scores(exp[0], title="Learning curve: Logistic regression (100 neurons)",
            xlabel="test set size (N)", ylabel="balanced accuracy")

plot_scores(exp[1], title="Learning curve: Logistic regression (500 neurons)",
            xlabel="test set size (N)", ylabel="balanced accuracy")

plot_scores(exp[2], title="Learning curve: Logistic regression (1000 neurons)",
            xlabel="test set size (N)", ylabel="balanced accuracy")


# ===== LOGISTIC REGRESSION: STIMULUS IDENTITY ===== #

scaler = StandardScaler()

symbol_logit = LogisticRegression(fit_intercept=True, multi_class="multinomial", max_iter=300, class_weight="balanced",
                                  solver="newton-cg", penalty="l2")

reg = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

exp = []

for i, x in enumerate([x100, x500, x1000]):

    print("Fit nr {}".format(i))

    N = x[0, :, :].shape[1]

    x_norm = scaler.fit_transform(x[0, :, :])  # (time_slice, trial, neuron): first dimension stores time window

    # fit logistic regression
    train_sc_obs, val_sc_obs = validation_curve(estimator=symbol_logit, X=x_norm, y=y_sym, cv=5, param_name="C",
                                                param_range=reg, scoring="balanced_accuracy")
    train_sc_rnd, val_sc_rnd = validation_curve(estimator=symbol_logit, X=x_norm, y=y_sym0_1, cv=5, param_name="C",
                                                param_range=reg, scoring="balanced_accuracy")
    train_sc_shf, val_sc_shf = validation_curve(estimator=symbol_logit, X=x_norm, y=y_sym0_2, cv=5, param_name="C", param_range=reg,
                                                scoring="balanced_accuracy")

    exp.append([["observed", train_sc_obs, val_sc_obs, reg],
                ["permuted", train_sc_rnd, val_sc_rnd, reg],
                ["shifted", train_sc_shf, val_sc_shf, reg]])

save(exp, "/project/3011085.04/snn/data/results/learning-curve_LR-symbol-100-500-1000.pkl")

plot_scores(exp[0], title="Validation curve (LR): symbol classification (100 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")

plot_scores(exp[1], title="Validation curve (LR): symbol classification (500 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")

plot_scores(exp[2], title="Validation curve (LR): symbol classification (1000 neurons)",
            xlabel="regularization parameter (C)", ylabel="balanced accuracy")

