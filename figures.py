
import sys, os, regex
import numpy as np
from util import load, Paths
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from plots import plot_scores

# ===== NETWORK TUNING ===== #

p = Paths()

rootdir = p.figures

if sys.argv[1] == "sequence-statistics":

    data = load("/project/3011085.04/snn/data/raw/seq_pairs-12k.pkl")

    current_palette = sns.color_palette("Blues", 6)
    sns.catplot(x="pairs", kind="count", data=pd.Series({"pairs":data}),
                color=current_palette[4])
    plt.title("Pair statistics (16k)")

    print("Saving {}".format(rootdir + "pair_statistics.png"))
    plt.savefig(rootdir + "pair_statistics.svg")
    plt.savefig(rootdir + "pair_statistics.png")

elif sys.argv[1] == "tuning":

    t100 = load("/project/3011085.04/snn/data/raw/tuning-100")
    t500 = load("/project/3011085.04/snn/data/raw/tuning-500")
    t1000 = load("/project/3011085.04/snn/data/raw/tuning_1000-B-0")

    p = [t1000]

    # make a plot

    plt.figure()
    for i, d in enumerate(p):

        plt.plot(p[i]["fRate-tune"][3], '--o', label="N = {}".format(1000))

    #plt.fill_between(x=np.arange(0, 6), y1=4.5, y2=5.5, color="gray", alpha=0.1)
    #plt.text(x=5, y=5.2, s="recurrent target range")
    plt.fill_between(x=np.arange(0, 6), y1=1.8, y2=2.2, color="gray", alpha=0.1)
    plt.text(x=0, y=2, s="input target range")
    plt.ylim(0, 5)
    plt.title("Tuning input connections")
    plt.ylabel("Mean firing rate (Hz)")
    plt.xlabel("iteration nr.")
    plt.legend(loc="best")

    print("Saving {}".format(rootdir + "input_tuning.png"))
    plt.savefig(rootdir + "input_tuning.svg")
    plt.savefig(rootdir + "input_tuning.png")

    plt.figure(figsize=(13, 5))
    for i, d in enumerate(p):

        plt.plot(p[i]["recurrent"][3], '--o', label="N = {}".format(p[i]["recurrent"][0]))

    #plt.fill_between(x=np.arange(0, 6), y1=4.5, y2=5.5, color="gray", alpha=0.1)
    #plt.text(x=5, y=5.2, s="recurrent target range")
    plt.fill_between(x=np.arange(0, 25), y1=4.5, y2=5.5, color="gray", alpha=0.3)
    #plt.ylim(0, 10)
    plt.text(x=0, y=6, s="recurrent target range")
    plt.title("Tuning recurrent connections")
    plt.ylabel("Mean firing rate (Hz)")
    plt.xlabel("iteration nr.")
    plt.legend(loc="best")

    print("Saving {}".format(rootdir + "recurrent_tuning.png"))
    plt.savefig(rootdir + "recurrent_tuning.svg")
    plt.savefig(rootdir + "recurrent_tuning.png")

elif sys.argv[1] == "subject-tuning":

    d = [f for f in os.listdir(p.raw) if regex.match(r'rates_1000-B-.*', f)]

    plt.figure()
    for f in d[:-1]:

        r = load(p.raw + "/" + f)

        # plot rates during tuning
        plt.plot(np.mean(r["fRate-tune"], 1), '--o')

    plt.show()


elif sys.argv[1] == "firing-rates":

    # shape = (subjects, tau, neurons)
    tmp = np.stack([load(p.raw + '/rates_1000-A-s{:02d}.pkl'.format(i+1))['fRate'] for i in np.arange(0, 10)])

    meanRates = np.mean(tmp, axis=2)
    means = np.mean(meanRates, axis=0)
    errs = np.std(meanRates, axis=0)

    # mean firing rates across subjects
    _, ax = plt.subplots()

    xval = np.array(['NA', 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

    ax.plot(meanRates.T[:, 1::], '-o', color='gray', alpha=0.3, zorder=1)
    ax.plot(meanRates.T[:, 0], '-o', color='gray', alpha=0.5, zorder=1, label="network instance (10)")
    ax.errorbar(x=np.arange(0, 9), y=means, yerr=errs, linewidth=2.5, zorder=2, label='mean, SD')
    ax.set_xticks(ticks=np.arange(0, 9))
    ax.set_xticklabels(labels=xval)
    ax.set_ylim((2, 8))
    ax.set_ylabel('network firing rate (Hz)')
    ax.set_xlabel('spike-rate adaptation time constant (sec)')
    ax.set_title('Firing rates (N = 1000)')
    ax.legend(loc='best')

    plt.savefig(p.figures + '/firing_rates.png')
    plt.savefig(p.figures + '/firing_rates.svg')

    # give an example of distribution (subject 5, gsra = 0.4)
    _, ax = plt.subplots()
    ax.hist(tmp[4, 4, :])


elif sys.argv[1] == "example-neurons":

    ds = load("/project/3011085.04/snn/data/interim/train_ds.pkl")
    net100 = load("/project/3011085.04/snn/data/raw/net-100.pkl")
    net500 = load("/project/3011085.04/snn/data/raw/net-500.pkl")
    net1000 = load("/project/3011085.04/snn/data/raw/net-1000.pkl")

    r100 = net100.avg_frate()
    r500 = net500.avg_frate()
    r1000 = net1000.avg_frate()

    # highly active neuron
    max_neuron = np.where(r100 == np.max(r100))[0][0]  # chose one
    min_neuron = np.where(r100 == np.min(r100))[0][0]  # chose one
    rnd_neuron = np.random.RandomState(1).randint(0, len(r100))

    titles = ["high-spiking neuron", "non-spiking neuron", "random neuron"]

    data = net100.recording["V"]
    time = np.arange(0, data.shape[-1]*data.shape[0])/1000 # express time in seconds
    onsets = np.arange(0, data.shape[0]*data.shape[-1], 51)
    sentences = np.sort(np.concatenate((np.where(ds.sequence == "1")[0], np.where(ds.sequence=="2")[0])))*51

    fig, ax = plt.subplots(4, 1, sharex="all", figsize=(14, 8))

    ymaxs = [-0.08, -0.04]
    ymins = [-0.1, -0.1]

    for i, n in enumerate([max_neuron, min_neuron, rnd_neuron]):

        d = np.hstack(data[0:40, n, :])
        d[np.where(d == -0.08)[0]-1] = -0.02  # plot spikes

        ax[i].plot(d, label="neuron nr. {:d}".format(n))
        ax[i].legend(loc="best")
        ax[i].set_title(titles[i])
        ax[i].set_ylabel("mem. potential \n (mV)")

    ax[3].vlines(onsets[0:40], ymax=0.5, ymin=0, label="stimulus onset")
    ax[3].vlines(sentences[0:6], ymax=0.5, ymin=0, color="red", label="sentence onset")
    ax[3].set_ylim(0, 2)
    ax[3].legend(loc="best")

    plt.xlabel("Time (msec)")
    plt.suptitle("Example neuron dynamics (N={:d})".format(data.shape[1]))

    print("Saving {}".format(rootdir + "example_neurons.png"))
    plt.savefig(rootdir + "example_neurons.svg")
    plt.savefig(rootdir + "example_neurons.png")

    pass

if sys.argv[1] == "validation-curve":

    N = [100, 500, 1000]

    for i, n in enumerate(N):

        scores = load(p.results + "/validation-curve_response-{}.pkl".format(n))

        plot_scores(scores, title="Validation curve: Logistic regression ({} neurons)".format(n),
                    xlabel="train set size (samples)", ylabel="balanced accuracy")

        print("Saving {}".format(rootdir + "validation_curve.png"))
        plt.savefig(rootdir + "validation_curve.svg")
        plt.savefig(rootdir + "validation_curve.png")

if sys.argv[1] == "learning-curve":

    N = [100, 500, 1000]

    for i, n in enumerate(N):

        scores = load(p.results + "/learning-curve_response-{}.pkl".format(n))

        plot_scores(scores, title="Learning curve: Logistic regression ({} neurons)".format(n),
                    xlabel="train set size (samples)", ylabel="balanced accuracy")

if sys.argv[1] == "adaptation-curve":

    d_res_obs = np.asarray(load(p.results + "/scores-1000-reset-observed.pkl"))
    d_nres_obs = np.asarray(load(p.results + "/scores-1000-noreset-observed.pkl"))

    d_res_noadapt_obs = np.asarray(load(p.results + "/scores-1000-reset-observed-noadapt.pkl"))
    d_res_noadapt_rnd = np.asarray(load(p.results + "/scores-1000-reset-observed-noadapt.pkl"))

    d_res_rnd = np.asarray(load(p.results + "/scores-1000-reset-permuted.pkl"))
    d_nres_rnd = np.asarray(load(p.results + "/scores-1000-noreset-permuted.pkl"))

    X = np.arange(0, 10)
    x_labels = np.array([50, 75, 100, 150, 200, 400, 500, 700, 1000, 1500])

    y1 = np.mean(d_res_obs, axis=1)
    y2 = np.mean(d_nres_obs, axis=1)
    y3 = np.mean(d_res_noadapt_obs, axis=1)

    y1std = np.std(d_res_obs, axis=1)
    y2std = np.std(d_nres_obs, axis=1)
    y3std = np.std(d_res_noadapt_obs, axis=1)

    rnd1 = np.mean(d_res_rnd, axis=1)
    rnd2 = np.mean(d_nres_rnd, axis=1)
    rnd1std = np.std(d_res_rnd, axis=1)
    rnd2std = np.std(d_nres_rnd, axis=1)

    lab1 = np.tile("reset", y1.shape[0])
    lab2 = np.tile("no reset", y1.shape[0])
    lab3 = np.tile("no adaptation", y1.shape[0])

    labs = pd.Series(np.hstack((lab1, lab2, lab3)))
    dat = pd.DataFrame(np.vstack((d_res_obs, d_nres_obs, d_res_noadapt_obs)),
                       columns=["fold-1", "fold-2", "fold-3", "fold-4", "fold-5"])
    dat["condition"] = labs
    dat["idx"] = np.arange(0, 21)
    datlong = pd.wide_to_long(df=dat, stubnames="fold-", i="idx", j="CV-fold")

    datlong = pd.melt(frame=dat, id_vars=["idx", "condition"], value_vars=["fold-1", "fold-2", "fold-3", "fold-4", "fold-5"])

    #plt.figure()
    #sns.lineplot(data=datlong, x=np.arange(0, 10), y="value", hue="condition")

    plt.fill_between(x=X, y1=y1+y1std, y2=y1-y1std, alpha=0.1)
    plt.fill_between(x=X, y1=y2+y2std, y2=y2-y2std, alpha=0.1)
    plt.plot(X, y1, '--o', label="reset")
    plt.plot(X, y2, '--o', label="noreset")
    plt.xticks(ticks=X, labels=x_labels)
    plt.ylabel("5-fold CV accuracy (prop. correct)")
    plt.xlabel("Neuronal adaptation time constant (msec)")
    plt.title("Neuronal adaptation and response readout")
    plt.legend(loc="best")
    plt.show()

    print("Saving {}".format(p.figures + "/adaptation_curve.png"))
    plt.savefig(p.figures + "/adaptation_curve.svg")
    plt.savefig(p.figures + "/adaptation_curve.png")

elif sys.argv[1] == "adaptation-response-decoding":

    plt.style.use('default')

    # Load in actual scores
    nsub = np.arange(0, 10)
    fig, ax = plt.subplots(2, 3, sharey='all', figsize=(18, 7))

    # loop over network size
    for k, N in enumerate([100, 500, 1000]):

        datatmp = [[] for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')]
        data = {k: None for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')}
        d = None

        # loop over subjects
        for i in np.arange(0, 10):

            subdata = load(p.results + "/scores_{}-A-observed-s{:02d}.pkl".format(N, (i+1)))
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

        # mean baseline
        # bavg = np.mean(base['test_balanced_accuracy'])
        # bstd = np.std(base['test_balanced_accuracy'])
        # bavg2 = np.mean(base['test_precision'])
        # bstd2 = np.std(base['test_precision'])

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

        plt.style.use("seaborn-paper")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # for k in ranged.shape[1]-1):
        ax[0, k].plot(d[:, 1::], '--o', alpha=0.5, color="gray", zorder=1)
        ax[0, k].plot(d[:, 0], '--o', color="gray", alpha=0.5, label="model instance (10)", zorder=1)
        ax[0, k].errorbar(x=np.arange(0, 9), y=avg, yerr=ers, linewidth=2.5, label="mean/SD", zorder=2)
        #ax[0].hlines(y=bavg, xmin=0, xmax=8, label="encoder", linestyles='-')
        #ax[0].fill_between(x=np.arange(0, 9), y1=bavg+bstd, y2=bavg-bstd, alpha=0.3, color=colors[0], zorder=1)

        #ax.hlines(y=bavg2, xmin=0, xmax=8, label="encoder", color=colors[1], linestyles='-')
        #ax.fill_between(x=np.arange(0, 9), y1=bavg2 + bstd2, y2=bavg2 - bstd2, alpha=0.3, color=colors[1], zorder=1)

        ax[0, k].set_xticks(ticks=np.arange(0, 9))
        ax[0, k].get_xaxis().set_ticklabels([])
        if k == 0:
            ax[0, k].set_ylabel("{} (percent)".format(score))
        ax[0, k].set_title("Neuronal adaptation for memory (N = {})".format(N))
        ax[0, k].legend(loc="best")

        # ax2 = ax.twinx()
        ax[1, k].errorbar(x=np.arange(0, 9), y=dpre_avg, yerr=dpre_ers, color=colors[0], linestyle='--', linewidth=2.5, label="precision (mean, SD)")
        ax[1, k].errorbar(x=np.arange(0, 9), y=drec_avg, yerr=drec_ers, color=colors[0], linestyle='-', linewidth=2.5, label="recall (mean, SD)")
        if k == 0:
            ax[1, k].set_ylabel("precision/recall (percent)")
        ax[1, k].set_xlabel("spike-rate adaptation (sec)")
        ax[1, k].set_xticks(ticks=np.arange(0, 9))
        ax[1, k].set_xticklabels(labels=xval)
        ax[1, k].legend(loc="best")
        #ax2.set_ylabel("precision/recall (proportion)")
        #ax2.tick_params(axis='y', labelcolor=colors[1])

    # plt.ylim(0.9, 1)
    plt.legend(loc='best')
    #plt.tight_layout()

    plt.savefig(p.figures + "/response-decoding_sharey.svg")
    plt.savefig(p.figures + "/response-decoding_sharey.png")

elif sys.argv[1] == 'response-decoding-dependency':

    # Load in actual scores
    nsub = np.arange(0, 10)
    datalist = []
    xval = np.array(['NA', 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

    for depth in [1, 2, 3, 4]:

        datatmp = [[] for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')]
        data = {k: None for k in ('balanced_accuracy', 'kappa', 'precision', 'recall')}
        d = None

        for i in np.arange(0, 10):

            subdata = load(p.results + "/scores_1000-A-observed-s{:02d}_{}.pkl".format((i + 1), depth))
            d1 = []
            d2 = []
            d3 = []
            d4 = []

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

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(1, 2)


    #----- PLOT -----#

    score = 'kappa'

    # shape = (subjects, tau, crossval)
    d = [np.mean(l['kappa'], axis=2).T for l in datalist]
    ers = [np.std(dat, axis=1) for dat in d]
    avg = [np.mean(dat, axis=1) for dat in d]

    ax[0].errorbar(x=np.arange(0, 9), y=avg[0], yerr=ers[0], linewidth=2.5, label='level 1')
    ax[0].errorbar(x=np.arange(0, 9), y=avg[1], yerr=ers[1], linewidth=2.5, label='level 2')
    ax[0].errorbar(x=np.arange(0, 9), y=avg[2], yerr=ers[2], linewidth=2.5, label='level 3')
    ax[0].errorbar(x=np.arange(0, 9), y=avg[3], yerr=ers[3], linewidth=2.5, label='level 4')

    ax[0].set_ylabel("kappa (prop. correct)")
    ax[0].set_xlabel("adaptation time constant (sec)")
    ax[0].set_xticks(ticks=np.arange(0, 9))
    ax[0].set_xticklabels(labels=xval)

    ax[0].legend(loc='best')

    # shape = (subjects, tau, crossval)
    d = [np.mean(l['precision'], axis=2).T for l in datalist]
    ers = [np.std(dat, axis=1) for dat in d]
    avg = [np.mean(dat, axis=1) for dat in d]


    ax[1].errorbar(x=np.arange(0, 9), y=avg[0], yerr=ers[0], linewidth=2.5, color=colors[0])
    ax[1].errorbar(x=np.arange(0, 9), y=avg[1], yerr=ers[1], linewidth=2.5, color=colors[1])
    ax[1].errorbar(x=np.arange(0, 9), y=avg[2], yerr=ers[2], linewidth=2.5, color=colors[2])
    ax[1].errorbar(x=np.arange(0, 9), y=avg[3], yerr=ers[3], linewidth=2.5, color=colors[3])

    # shape = (subjects, tau, crossval)
    d = [np.mean(l['recall'], axis=2).T for l in datalist]
    ers = [np.std(dat, axis=1) for dat in d]
    avg = [np.mean(dat, axis=1) for dat in d]

    ax[1].errorbar(x=np.arange(0, 9), linestyle='--', y=avg[0], yerr=ers[0], linewidth=3, color=colors[0], label='level 1')
    ax[1].errorbar(x=np.arange(0, 9), linestyle='--', y=avg[1], yerr=ers[1], linewidth=3, color=colors[1], label='level 2')
    ax[1].errorbar(x=np.arange(0, 9), linestyle='--', y=avg[2], yerr=ers[2], linewidth=3, color=colors[2], label='level 3')
    ax[1].errorbar(x=np.arange(0, 9), linestyle='--', y=avg[3], yerr=ers[3], linewidth=3, color=colors[3], label='level 4')

    ax[1].set_ylabel("precision/recall (prop.)")
    ax[1].set_xlabel("adaptation time constant (sec)")
    ax[1].set_xticks(ticks=np.arange(0, 9))
    ax[1].set_xticklabels(labels=xval)

    plt.suptitle("Accuracy by dependency length")

    # save
    plt.savefig(p.figures + "/response-decoding-dependency.svg")
    plt.savefig(p.figures + "/response-decoding-dependency.png")

    #----- SHAPE DATAFRAME -----#

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
    ax = sns.catplot(x='level', y='kappa', hue='tau', palette='ch:.25', kind='bar', data=df, legend_out=True)
    ax.set(xlabel='distance', xticklabels=['1', '2', '3', '4'])
    ax._legend.set_title('tau (sec)')
    ax.savefig(p.figures + '/response-decoding-dependency-bar.png')
    ax.savefig(p.figures + '/response-decoding-dependency-bar.svg')

elif sys.argv[1] == "stimulus-buildup":

    d = np.asarray(load(p.results + "/stimulus-decoding_old-observed_s01.pkl"))
    d0 = np.asarray(load(p.results + "/stimulus-decoding_new-observed_s01.pkl"))

    dm = np.mean(d, axis=1)
    dm0 = np.mean(d0, axis=1)
    ds = np.std(d, axis=1)
    ds0 = np.std(d0, axis=1)

    xval = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    plt.figure()
    plt.errorbar(x=np.arange(0, 5), y=dm, yerr=ds, linewidth=1, label="mean (sd)", zorder=1)
    #plt.errorbar(x=np.arange(0, 5), y=dm0, yerr=ds0, linewidth=1, label="mean (sd)", zorder=2)
    plt.xticks(ticks=np.arange(0, 5), labels=xval)
    plt.ylim(0, 1.1)
    plt.legend(loc="best")
    plt.show()
