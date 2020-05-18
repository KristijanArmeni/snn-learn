
import numpy as np
from util import load, Paths
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# ===== LOAD DATA ===== #
p = Paths()

# Load in actual scores
nsub = np.arange(0, 10)
datalist = []
xval = np.array(['NA', 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

for depth in [1, 2, 3, 4]:

    datatmp = [[] for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')]
    data = {k: None for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')}
    d = None

    # loop for each subject
    for i in np.arange(0, 10):

        subdata = load(p.results + "/newtime/scores_1000-A-observed-s{:02d}_{}.pkl".format((i + 1), depth))
        d1 = []
        d2 = []
        d3 = []
        d4 = []

        # for each dependency length
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

    datalist.append(data)

# shape = (subjects, tau, crossval)
d = [np.mean(l['kappa'], axis=2).T for l in datalist]

# ===== SHAPE DATAFRAME ===== #

plt.style.use("default")
df = pd.DataFrame(np.zeros(shape=(4*10*9, 6)), columns=['subject', 'tau', 'level', 'kappa', 'precision', 'recall'])

dat = np.stack(d).reshape((4*9*10))

i = 0
for subject in np.arange(0, 10):
    for tau_i, tau in enumerate(xval):
        for level in np.arange(0, 4):

            df.loc[i, 'subject'] = subject
            df.loc[i, 'tau'] = tau
            df.loc[i, 'level'] = level + 1
            df.loc[i, 'kappa'] = d[level][tau_i, subject]

            i = i+1

df['tau'].astype('category')
df['level'].astype('category')

# ===== DRAW THE PLOT ===== #

clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns.set(font_scale=1.3)
sns.set_style("ticks")
fig, ax = plt.subplots(1, 1, sharey='all', figsize=(11, 6))

sns.stripplot(x='tau', y='kappa', data=df, hue="level", palette="GnBu_d", alpha=0.8, size=9, linewidth=1, edgecolor="gray", ax=ax)
sns.pointplot(x='tau', y='kappa', data=df, ci="sd", hue="level", palette="GnBu_d", linestyles="--", ax=ax, join=True, edgecolor="gray")

handles, labels = ax.get_legend_handles_labels()
labels = [int(float(s)) for s in labels]

ax.set_ylabel("decoding performance\n(kappa)")
ax.set_xlabel("tau (sec)")
plt.legend(handles[0:4], labels[0:4], loc="best", title="Sequence position")
sns.despine()

plt.savefig(p.figures + '/response-decoding-dependency-single.png')
plt.savefig(p.figures + '/response-decoding-dependency-single.svg')
