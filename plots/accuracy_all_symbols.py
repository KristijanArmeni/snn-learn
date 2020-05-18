import numpy as np
from util import load, Paths
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

p = Paths()

plt.style.use('seaborn-talk')

# Load in actual scores
nsub = np.arange(0, 10)
N = 1000  # number of neurons in the model

datatmp = [[] for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')]
data = {k: None for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')}
d = None

# loop over subjects
for i in np.arange(0, 10):

    subdata = load(p.results + "/newtime/scores_{}-A-observed-s{:02d}_all.pkl".format(N, i+1))
    d1 = []
    d2 = []
    d3 = []
    d4 = []

    # loop over tau values
    for j in range(len(subdata)):

        d1.append(subdata[j]['test_balanced_accuracy'])
        d2.append(subdata[j]['test_kappa'])
        d3.append(subdata[j]['test_precision'])
        d4.append(subdata[j]['test_recall'])

    datatmp[0].append(np.asarray(d1))
    datatmp[1].append(np.asarray(d2))
    datatmp[2].append(np.asarray(d3))
    datatmp[3].append(np.asarray(d4))

data['balanced_accuracy'] = np.asarray(datatmp[0])
data['kappa'] = np.asarray(datatmp[1])
data['precision'] = np.asarray(datatmp[2])
data['recall'] = np.asarray(datatmp[3])

# ===== SHAPE DATAFRAME ===== #

plt.style.use("default")
df = pd.DataFrame(np.zeros(shape=(10*9, 4)), columns=['subject', 'tau', 'metric', 'score'])

# these are tau values
xval = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

i = 0
for subject in np.arange(0, 10):

    for tau_i, tau in enumerate(xval):

        for metric in ["kappa", "precision", "recall"]:

            df.loc[i, 'subject'] = subject
            df.loc[i, 'tau'] = tau
            df.loc[i, 'metric'] = metric
            df.loc[i, 'score'] = np.mean(data[metric][subject][tau_i, :])

            i = i+1

df['tau'].astype('category')

# ===== PLOT ===== #

sns.set(palette='GnBu_d')
sns.set(font_scale=1.3)
sns.set_style("ticks")
fig, ax = plt.subplots(1, 1, sharey='all', figsize=(11, 6))

data = df[df.metric == 'kappa']
sns.lineplot(data=data, x="tau", y="score", units="subject", estimator=None, palette='GnBu_d',
             marker="o", markersize=9, markeredgecolor="white", linestyle='-.')

ax.set_ylabel("decoding performance\n(kappa)")
ax.set_xlabel("tau (sec)")
sns.despine()

plt.savefig(p.figures + '/response-decoding-all.png')
plt.savefig(p.figures + '/response-decoding-all.svg')

#----- PLOT -----#

score = 'kappa'

# shape = (subjects, tau, crossval)
d = np.mean(data[score], axis=2).T
ers = np.std(d, axis=1)
avg = np.mean(d, axis=1)

# precision/recall
dpre_avg = np.mean(np.mean(data['precision'], axis=2), axis=0)
dpre_ers = np.std(np.mean(data['precision'], axis=2), axis=0)
drec_avg = np.mean(np.mean(data['recall'], axis=2), axis=0)
drec_ers = np.std(np.mean(data['recall'], axis=2), axis=0)

xval = np.array(['NA', 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

#plt.style.use("seaborn-paper")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()
# for k in ranged.shape[1]-1):
ax.plot(d[:, 1::], '--o', alpha=0.5, color="gray", zorder=1)
ax.plot(d[:, 0], '--o', color="gray", alpha=0.5, label="model instance (10)", zorder=1)
ax.errorbar(x=np.arange(0, 9), y=avg, yerr=ers, linewidth=2.5, label="mean/SD", zorder=2)
#ax[0].hlines(y=bavg, xmin=0, xmax=8, label="encoder", linestyles='-')
#ax[0].fill_between(x=np.arange(0, 9), y1=bavg+bstd, y2=bavg-bstd, alpha=0.3, color=colors[0], zorder=1)

#ax.hlines(y=bavg2, xmin=0, xmax=8, label="encoder", color=colors[1], linestyles='-')
#ax.fill_between(x=np.arange(0, 9), y1=bavg2 + bstd2, y2=bavg2 - bstd2, alpha=0.3, color=colors[1], zorder=1)

ax.set_xticks(ticks=np.arange(0, 9))
ax.get_xaxis().set_ticklabels([])
#if k == 0:
    # ax[0, k].set_ylabel("test {} score\n(cross-validated)".format(score))
ax.tick_params(axis='y', which='major', labelsize=14)
#ax[0, k].set_title("Neuronal adaptation for memory (N = {})".format(N))
ax.legend(loc="best")

# ax2 = ax.twinx()
ax.errorbar(x=np.arange(0, 9), y=dpre_avg, yerr=dpre_ers, color=colors[0], linestyle='--', linewidth=2.5, label="precision (mean, SD)")
ax.errorbar(x=np.arange(0, 9), y=drec_avg, yerr=drec_ers, color=colors[0], linestyle='-', linewidth=2.5, label="recall (mean, SD)")
#if k == 0:
    #ax[1, k].set_ylabel("test precision/recall score\n(cross-validated)")
#ax[1, k].set_xlabel("spike-rate adaptation (sec)")
ax.set_xticks(ticks=np.arange(0, 9))
ax.set_xticklabels(labels=xval)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(loc="best")
#ax2.set_ylabel("precision/recall (proportion)")
#ax2.tick_params(axis='y', labelcolor=colors[1])

# plt.ylim(0.9, 1)
plt.legend(loc='best')
#plt.tight_layout()

plt.savefig(p.figures + "/response-decoding_sharey.svg")
plt.savefig(p.figures + "/response-decoding_sharey.png")
