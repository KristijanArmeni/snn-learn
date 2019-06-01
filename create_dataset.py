
# DATA MANIPULATION
import numpy as np
import sys
from stimuli import enumerate_sequence, Dataset
from util import save, Paths

p = Paths()

n_inner = np.arange(1, 5)

seq, rsp, inp, seq_pairs, n_outer = enumerate_sequence(lengths=n_inner, n_target=12000, step=100, seed=95)
dataset = Dataset(sequence=seq, response=rsp, encoding=inp, pairs=seq_pairs)  # inherit the torch Dataset class properties

print("Saving to " + p.raw)
save(dataset, p.raw + "/12ax-12k.pkl")
