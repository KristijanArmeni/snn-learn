
# OWN MODULES
from util import save, load, Paths
import numpy as np

p = Paths()

# load the sequence
ds = load(p.raw + "/12ax-12k.pkl")

# use the segment method
seq = ds.segment(return_unique=True, reshape_back=False)

# grab the first tuple (== list of sentences)
s = seq[0]

# concatenate lists into strings
s2 = ["".join(i) for i in s]

# test if the lengths are the same
assert len(s2) == len(np.unique(np.asarray(s2)))

# now do a test for one where I double one string
snot = ["".join(i) for i in seq[0]]

# grab a random sequence and double it
snot.append(snot[1313])

assert len(snot) == len(np.unique(np.asarray(snot)))