import os
from util import load, Paths
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

data = load("/project/3011085.04/snn/data/raw/seq_pairs-12k.pkl")

p = Paths()

rootdir = p.figures

ncount = len(data)

current_palette = sns.color_palette("Blues", 6)

sns.set_style('white')
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(x="pairs", data=pd.Series({"pairs": data}),
            color=current_palette[4], ax=ax)
#plt.tick_params(axis='both', which='major', labelsize='14')

# following answer here:
# https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies
ax2 = ax.twinx()

ax2.yaxis.tick_left()
ax.yaxis.tick_right()

for p in ax.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),
            ha='center', va='bottom', size=14)  # set the alignment of the text

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0, 100/2)
ax.set_ylim(0, ncount/2)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# remove the labels
ax.set_ylabel('')
ax2.set_ylabel('')

ax.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)

print("Saving {}".format(os.path.join(rootdir, "pair_statistics.png")))
plt.savefig(os.path.join(rootdir, "pair_statistics.svg"))
plt.savefig(os.path.join(rootdir, "pair_statistics.png"))
