
import sys, os, regex
import numpy as np
from util import load, Paths
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from plots import plot_scores
plt.style.use('seaborn-notebook')

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


elif sys.argv[1] == "firing rate":

    net100 = load("/project/3011085.04/snn/data/raw/net-100.pkl")
    net500 = load("/project/3011085.04/snn/data/raw/net-500.pkl")
    net1000 = load("/project/3011085.04/snn/data/raw/net-1000.pkl")

    r100 = net100.avg_frate()
    r500 = net500.avg_frate()
    r1000 = net1000.avg_frate()

    rates = [r100, r500, r1000]
    N = [100, 500, 1000]

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    for i, data in enumerate(rates):

        ax[i].hist(data, edgecolor="white")
        _, ymax = ax[i].get_ylim()
        ax[i].vlines(x=np.mean(data), ymin=0, ymax=ymax/2, linestyles='--')
        ax[i].set_title("N = {}".format(N[i]))
        if i == 0:
            ax[i].set_ylabel("Count (neurons)")
        if i == 1:
            ax[i].set_xlabel("Mean firing rate (Hz)")

    plt.suptitle("Neuronal firing rates")
    print("Saving {}".format(rootdir + "firing_rates.png"))
    plt.savefig(rootdir + "firing_rates.svg")
    plt.savefig(rootdir + "firing_rates.png")

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

elif sys.argv[1] == "stimulus-buildup":

    d = np.asarray(load(p.results + "/stimulus-decoding_old-observed_s01.pkl"))
    d0 = np.asarray(load(p.results + "/stimulus-decoding_old-permuted_s01.pkl"))

    dm = np.mean(d, axis=0)
    dm0 = np.mean(d0, axis=0)
    ds = np.std(d, axis=0)
    ds0 = np.std(d0, axis=0)

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    plt.figure()
    plt.fill_between(x=x, y1=dm + ds, y2=dm - ds, alpha=0.1)
    plt.fill_between(x=x, y1=dm0 + ds0, y2=dm0 - ds0, alpha=0.1)
    plt.plot(x, dm, '--o', label="observed")
    plt.plot(x, dm0, '--o', label="permuted")
    plt.ylim(0, 1.1)
    plt.legend(loc="best")
    plt.show()