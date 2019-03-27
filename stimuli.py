
import numpy as np
from torch.utils import data


def generate_sequence(n_inner, n_outer, seed=None):

    """ generate_sequence() creates a sequence of symbols to be used in
    1-2-AX continous performance task as described in Frank et al., 2001
    (doi: 10.3758/CABN.1.2.137).\n

    USE AS
    sequence, response = generate_sequence(n_inner, n_outer)

    INPUTS
    n_inner = integer, maximal number of symbol pairs in inner loops of the sequence, 1:1:n_inner
    n_outer = integer, number of outer loops in the sequence
    seed    = boolean or int, for deterministic random number generation

    OUTPUTS
    sequence = numpy array, final sequence to be used
    response = boolean array, correct responses to items in the sequence
    """
    pass

    possible_lens = np.arange(1, n_inner+1)  # len(inner_loop) = {1, 2, 3, 4}

    # Define alphabet
    outer_symbols = ["1", "2"]        # Sampling of outer symbols has equal probability
    inner_symbols = ["A", "B", "C"]
    target_symbols = ["X", "Y", "Z"]

    # Define symbols combinations
    combinations = np.array(np.meshgrid(inner_symbols, target_symbols)).T.reshape(-1, 2)
    allpairs = ["".join(row) for row in combinations]
    possible_targets = ["AX", "BY"]

    # Define sequences
    inner_seq = np.random.RandomState(seed).choice(possible_lens, n_outer)              # determine lengths of inner loops
    outer_seq = np.random.RandomState(seed).choice(outer_symbols, n_outer, [0.5, 0.5])  # choose 1 or 2 with p = 0.5

    # LOOP

    out_seq = np.array([], dtype=str)

    # Outer loop
    for i in np.arange(0, len(outer_seq)):

        current = outer_seq[i]  #
        current_n_inner = inner_seq[i]

        out_seq = np.append(out_seq, current)

        inn_seq = np.array([], dtype=str)

        if seed is not None:
            seed = seed+5*i  # every outer loop a different seed

        opt1 = np.random.RandomState(seed).choice(a=allpairs, size=current_n_inner)  # randomly pick any possible combination
        opt2 = np.random.RandomState(seed).choice(a=possible_targets, size=current_n_inner, p=[0.5, 0.5])  # pick 'AX' or 'BY' with p = 0.5

        # Inner loop generation routine
        for j in np.arange(0, current_n_inner):

            if seed is not None:
                seed = seed+5+i*j  # Make sure seed is outer loop and inner loop specific

            # pick a random or a target pair with p=0.5
            inn_seq = np.append(inn_seq, np.random.RandomState(seed).choice(a=[opt1[j], opt2[j]], size=1, p=[0.5, 0.5]))

        out_seq = np.append(out_seq, inn_seq)

    # Create sequence where each item has length 1
    sequence = np.concatenate(np.array([list(item) for item in out_seq]))

    # Determine correct responses
    response = np.ndarray(sequence.shape, dtype=bool)

    for h in np.arange(0, len(response)):

        current_symbol = sequence[h]
        if h > 0:
            prev_symbol = sequence[h-1]

        if current_symbol in outer_symbols:
            last_number = current_symbol
            response[h] = 0
        elif current_symbol in inner_symbols:
            response[h] = 0
        elif current_symbol in target_symbols:
            if last_number == "1" and prev_symbol == "A" and current_symbol == "X":
                response[h] = 1
            elif last_number == "2" and prev_symbol == "B" and current_symbol == "Y":
                response[h] = 1
            else:
                response[h] = 0

    return sequence, response, out_seq


def encode_input(sequence):
    """encode_input() create a matrix of orthonormal vectors
    representing symbols in the input sequence

    USE AS
    inp = encode_input(sequence)

    INPUTS
    sequence = 1D numpy string array, representing input symbol sequence"""

    alphabet = ["1", "2", "A", "B", "C", "X", "Y", "Z"]
    dim = len(alphabet)
    inp = np.zeros(shape=(dim, sequence.shape[0]), dtype=int)
    vec = np.eye(dim, dim)

    for i in np.arange(0, len(sequence), 1):

        inp[:, i] = vec[np.isin(alphabet, sequence[i])]

    return inp


class Dataset(data.Dataset):

    def __init__(self, sequence=None, response=None, encoding=None):

        self.sequence = sequence
        self.response = response
        self.encoding = encoding

    def __getitem__(self, index):

        symbol = self.sequence[index]
        response = self.response[index]
        vector = self.encoding[:, index]

        return symbol, response, vector

    def __len__(self):

        return len(self.sequence)
