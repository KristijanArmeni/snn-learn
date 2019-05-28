
# DATA MANIPULATION
import numpy as np
import sys
from stimuli import enumerate_sequence, Dataset
from util import save, load

n_inner = np.arange(1, 5)

seq, rsp, inp, seq_pairs, n_outer = enumerate_sequence(lengths=n_inner, n_target=2500, step=20, seed=95)
dataset = Dataset(sequence=seq, response=rsp, encoding=inp)  # inherit the torch Dataset class properties

if sys.argv[1] == "full":
    save(dataset, sys.argv[1] + "/stimuli_{}-{}.pkl".format(n_inner[0], n_inner[-1]))


# ===== SPLIT THE DATASET =====#

train_idx = np.arange(0, int(len(dataset)*0.6))                   # 60 % training set
test_idx = np.arange(int(len(dataset)*0.6), int(len(dataset)*1))  # 40 % training set

train_ds = Dataset(sequence=dataset[train_idx][0],
                   response=dataset[train_idx][1],
                   encoding=dataset[train_idx][2])

test_ds = Dataset(sequence=dataset[test_idx][0],
                  response=dataset[test_idx][1],
                  encoding=dataset[test_idx][2])

print("Saving to /project/3011085.04/snn/data/interim/")
save(train_ds, "/project/3011085.04/snn/data/interim/train_ds.pkl")
save(test_ds, "/project/3011085.04/snn/data/interim/test_ds.pkl")