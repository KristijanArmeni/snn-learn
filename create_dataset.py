
# DATA MANIPULATION
import numpy as np
import sys
from stimuli import enumerate_sequence, Dataset
from util import save, load

n_inner = np.arange(1, 5)

seq, rsp, inp, seq_pairs, n_outer = enumerate_sequence(lengths=n_inner, n_target=2500, step=20, seed=95)
dat = Dataset(sequence=seq, response=rsp, encoding=inp)  # inherit the torch Dataset class properties

save(dat, sys.argv[1] + "/stimuli_{}-{}.pkl".format(n_inner[0], n_inner[-1]))
