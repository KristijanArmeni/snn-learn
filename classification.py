# ===== CLASSIFICATION ANALYSIS ===== #

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.svm import SVC

# load data
net = load("/project/3011085.04/snn/data/raw/recording.pkl")


# load the actual data
x = load("/project/3011085.04/snn/data/interim/average_states.pkl")
y = load("/project/3011085.04/snn/data/interim/responses.pkl")


# create fake data
x2 = np.array(x)
x2[y] += 0.03  # add systematic average increase of 0.03 mV
save(x2, "/project/3011085.04/snn/data/interim/average_states_constructed.pkl")

plt.figure()
plt.hist(x.flatten())
plt.figure()
plt.hist(x2.flatten())

# random responses
rand_y = np.random.RandomState(55).permutation(y)
save(rand_y, "/project/3011085.04/snn/data/interim/responses_surrogate.pkl")


# plot distributions
yplot = np.zeros(y.shape, dtype=int)
yplot[np.where(y)] = 1
plt.figure()
plt.hist(yplot)

randyplot = np.zeros(rand_y.shape, dtype=int)
randyplot[np.where(rand_y)] = 1
plt.figure()
plt.hist(randyplot)

# Learning curve: Naive bayes (imbalanced classes)

N, train_lc, val_lc = learning_curve(estimator=GaussianNB(), X=x, y=y, cv=5, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_surr, val_lc_surr = learning_curve(estimator=GaussianNB(), X=x, y=rand_y, cv=5, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_fake, val_lc_fake = learning_curve(estimator=GaussianNB(), X=x2, y=y, cv=5, train_sizes=np.linspace(0.1, 1, 7))

scoresNBim = [["observed response", train_lc, val_lc, N],
          ["permuted response (null)", train_lc_surr, val_lc_surr, N],
          ["constructed feature", train_lc_fake, val_lc_fake, N]]

save(scoresNBim, "/project/3011085.04/snn/data/results/scoresNB_imbalanced.pkl")

plot_scores(scoresNBim, title="Learning curve: Naive Bayes (5-fold cv, imbalanced classes)")

# Learning curve: SVC (imbalanced classes)

svc = SVC(kernel="linear", C=1e10)

N, train_lc, val_lc = learning_curve(estimator=svc, X=x, y=y, cv=5, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_surr, val_lc_surr = learning_curve(estimator=svc, X=x, y=rand_y, cv=5, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_fake, val_lc_fake = learning_curve(estimator=svc, X=x2, y=y, cv=5, train_sizes=np.linspace(0.1, 1, 7))

scoresNBim = [["observed response", train_lc, val_lc, N],
          ["permuted response (null)", train_lc_surr, val_lc_surr, N],
          ["constructed feature", train_lc_fake, val_lc_fake, N]]

save(scoresNBim, "/project/3011085.04/snn/data/results/scoresSVC_imbalanced.pkl")

plot_scores(scoresNBim, title="Learning curve: SVC (5-fold cv, imbalanced classes)")

# Equal the number of training samples
sel_false = np.random.RandomState(55).choice(np.where(y == False)[0], replace=False, size=np.sum(y))
sel = np.sort(np.concatenate((sel_false, (np.where(y == True)[0]))))

rand_y2 = np.random.RandomState(55).permutation(y[sel])

x2 = np.array(x[sel, :])
x2[y[sel]] += 0.03  # add systematic average increase of 0.03 mV to membrane states

# Define the cross-validation scheme

cv = StratifiedKFold(n_splits=5, shuffle=False)

N, train_lc, test_lc = learning_curve(estimator=GaussianNB(), X=x[sel, :], y=y[sel], cv=cv, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_surr, test_lc_surr = learning_curve(estimator=GaussianNB(), X=x[sel, :], y=rand_y2, cv=cv, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_fake, test_lc_fake = learning_curve(estimator=GaussianNB(), X=x2, y=y[sel], cv=cv, train_sizes=np.linspace(0.1, 1, 7))

scoresNBb = [["observed response", train_lc, test_lc, N],
          ["permuted response (null)", train_lc_surr, test_lc_surr, N],
          ["constructed feature", train_lc_fake, test_lc_fake, N]]

# save output
save(scoresNBb, "/project/3011085.04/snn/data/results/scoresNB_balanced.pkl")

plot_scores(scoresNBb, title="Learning curves: Naive Bayes (5-fold cross-validation, balanced classes)")

# ==== SVM ===== #

svc = SVC(kernel="linear", C=1e10)

N, train_lc, test_lc = learning_curve(estimator=svc, X=x[sel, :], y=y[sel], cv=cv, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_surr, test_lc_surr = learning_curve(estimator=svc, X=x[sel, :], y=rand_y2, cv=cv, train_sizes=np.linspace(0.1, 1, 7))
N, train_lc_fake, test_lc_fake = learning_curve(estimator=svc, X=x2, y=y[sel], cv=cv, train_sizes=np.linspace(0.1, 1, 7))

scoresSVC = [["observed response", train_lc, test_lc, N],
            ["permuted response (null)", train_lc_surr, test_lc_surr, N],
            ["constructed feature", train_lc_fake, test_lc_fake, N]]

# save output
save(scoresSVC, "/project/3011085.04/snn/data/results/scoresSVC_balanced.pkl")

plot_scores(scoresSVC, title="Learning curves: SVM (5-fold cross-validation, balanced classes)")


plt.figure()
plt.imshow(net.recording["V"][0,:,:], aspect="auto")
plt.xlabel("time (sec)")
plt.title("Trial {}".format(0+1))

plt.figure()
plt.imshow(net.recording["spikes"][1,:,:], aspect="auto")
plt.xlabel("time (sec)")
plt.title("Trial {}".format(1+1))

# ===== DEMO ===== #

seq, rsp, out_seq = generate_sequence(n_inner=4, n_outer=5, seed=95)
inp = encode_input(sequence=seq)
ds = Dataset(sequence=seq, response=rsp, encoding=inp)


# Initialize parameters
parameters = Params(tmin=0, tmax=0.5, dt=0.001,   # simulation parameters
                    tau_gsra=0.4, tau_gref=0.002)  # synaptic adaptation

# initialize network instance
net_demo = SNN(params=parameters, n_neurons=10, input_dim=8, output_dim=2, syn_adapt=False)
net_demo.config_input_weights(mean=0.4, density=0.5, seed=1)
net_demo.config_recurrent_weights(density=0.5, ex=0.8, seed=2)

# scaling
net_demo.w["input"] *= 2.1
net_demo.w["recurrent"] *= 3.7e-9

net_demo.config_recording(n_neurons=net_demo.neurons["N"], t=parameters.sim["t"], dataset=ds, downsample=None)

input_current = parameters.step_current(t=net_demo.recording["t_orig"], on=0, off=0.5, amp=4.8e-9)

net_demo.w["recurrent"][:] = 0  # disconnect recurrent weights
net_demo.train(dataset=ds, current=input_current, reset_states="sentence")

plot_trial(model=net_demo, trial=2, variable="V")
plot_time(model=net_demo, trial=2, variable="V")