import sys, os, regex
import numpy as np
from util import load, Paths
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from plots import plot_scores

if sys.argv[1] == "sequence-statistics":

    data = load("/project/3011085.04/snn/data/raw/seq_pairs-12k.pkl")

    current_palette = sns.color_palette("Blues", 6)
    sns.catplot(x="pairs", kind="count", data=pd.Series({"pairs":data}),
                color=current_palette[4])
    plt.tick_params(axis='both', which='major', labelsize='14')
    #plt.title("Pair statistics (16k)")

    print("Saving {}".format(rootdir + "pair_statistics.png"))
    plt.savefig(rootdir + "pair_statistics.svg")
    plt.savefig(rootdir + "pair_statistics.png")