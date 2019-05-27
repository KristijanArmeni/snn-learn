
import sys
import numpy as np
from util import load, save
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

# ===== NETWORK TUNING ===== #

rootdir = "/project/3011085.04/snn/data/figures/"

if sys.argv[1] == "tuning":

    t100 = load("/project/3011085.04/snn/data/raw/tuning-100")
    t500 = load("/project/3011085.04/snn/data/raw/tuning-500")
    t950 = load("/project/3011085.04/snn/data/raw/tuning-950")

    p = [t100, t500, t950]

    # make a plot

    plt.figure()
    plt.plot(p[0]["recurrent"][3], '--o', label="N = {}".format(p[0]["recurrent"][0]))
    plt.plot(p[1]["recurrent"][3], '--o', label="N = {}".format(p[1]["recurrent"][0]))
    plt.plot(p[2]["recurrent"][3], '--o', label="N = {}".format(p[2]["recurrent"][0]))

    plt.fill_between(x=np.arange(0, 17), y1=4.5, y2=5.5, color="gray", alpha=0.1)
    plt.text(x=22, y=5.2, s="recurrent target range")
    plt.fill_between(x=np.arange(0, 17), y1=1.8, y2=2.2, color="gray", alpha=0.1)
    plt.text(x=22, y=2, s="input target range")
    plt.ylim(0, 15)
    plt.title("Tuning recurrent connections")
    plt.ylabel("Mean firing rate (Hz)")
    plt.xlabel("iteration nr.")
    plt.legend(loc="best")

    print("Saving {}".format(rootdir + "recurrent_tuning.png"))
    plt.savefig(rootdir + "recurrent_tuning.svg")
    plt.savefig(rootdir + "recurrent_tuning.png")


