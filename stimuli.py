
import numpy as np
from torch.utils import data


def generate_sequence(possible_n_inner=None, n_outer=None, seed=None):

    """ generate_sequence() creates a sequence of symbols and responses for the
    1-2-AX continous performance task as described in Frank et al., 2001
    (doi: 10.3758/CABN.1.2.137).\n

    EXAMPLE USAGE:
    sequence, response, out_sequence = generate_sequence(possible_n_inner=np.arange(1, 5), n_outer=50, seed=123)

    INPUTS
    possible_n_inner = int, maximal number of symbol pairs in inner loops of the sequence, 1:1:n_inner
    n_outer          = int, number of outer loops in the sequence
    seed (optional)  = boolean, for deterministic random number generation (default = None)

    OUTPUTS
    sequence =      numpy array, final sequence to be used
    response =      boolean array, correct responses to items in the sequence
    out_sequence =  nd array, output sequence where each element is a symbol pair, useful for checking statistics
    """

    # Define alphabet
    outer_symbols = ["1", "2"]        # Sampling of outer symbols has equal probability
    inner_symbols = ["A", "B", "C"]
    target_symbols = ["X", "Y", "Z"]

    # Define symbols combinations
    combinations = np.array(np.meshgrid(inner_symbols, target_symbols)).T.reshape(-1, 2)
    allpairs = ["".join(row) for row in combinations]
    possible_targets = ["AX", "BY"]

    # Define sequences
    inner_seq = np.random.RandomState(seed).choice(possible_n_inner, n_outer, replace=True)                # determine lengths of inner loops
    outer_seq = np.random.RandomState(seed+123).choice(outer_symbols, n_outer, replace=True, p=[0.5, 0.5])  # choose 1 or 2 with p = 0.5

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

        opt1 = np.random.RandomState(seed).choice(a=allpairs, size=current_n_inner)  # array with selection for any symbol pair
        opt2 = np.random.RandomState(seed+123).choice(a=possible_targets, size=current_n_inner, p=[0.5, 0.5])  # pick 'AX' or 'BY' with p = 0.5

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


def enumerate_sequence(lengths=(1, 4), n_target=100, step=5, seed=95):

    """

    :param n_target:  double, lower bound on the number of symbols in the sequence
    :param :
    :return:
    """

    n_outer = 1
    n_unique = 0
    c = 0
    # generate sequence here
    while n_unique < n_target:

        n_outer += step
        c += 1

        print("[{:d}] Trying n = {:d}".format(c, n_outer))

        s, r, s_pairs = generate_sequence(possible_n_inner=lengths, n_outer=n_outer, seed=seed)
        d = Dataset(sequence=s, response=r, encoding=encode_input(sequence=s), pairs=s_pairs)

        seq, rsp, inp, pairs = d.segment(return_unique=True, reshape_back=True)

        n_unique = seq.shape[0]  # number of symbols when sentences are unique
        print("Number of symbols:{}".format(n_unique))

    return seq, rsp, inp, pairs, n_outer


class Dataset(data.Dataset):

    def __init__(self, sequence=None, response=None, encoding=None, pairs=None):

        self.sequence = sequence
        self.response = response
        self.encoding = encoding
        self.pairs = pairs

    def __getitem__(self, index):

        symbol = self.sequence[index]
        response = self.response[index]
        vector = self.encoding[:, index]

        return symbol, response, vector

    def __len__(self):

        return len(self.sequence)

    def segment(self, return_unique=False, reshape_back=False):

        id = np.sort(np.hstack([np.where(self.sequence == "1"), np.where(self.sequence == "2")]))
        id_pairs = np.sort(np.hstack([np.where(self.pairs == "1"), np.where(self.pairs == "2")]))

        sentmp = np.split(ary=self.sequence, indices_or_sections=id[0])
        rsptmp = np.split(ary=self.response, indices_or_sections=id[0])
        inptmp = np.split(ary=self.encoding, indices_or_sections=id[0], axis=1)
        pairstmp = np.split(ary=self.pairs, indices_or_sections=id_pairs[0])

        sen = [a.tolist() for a in sentmp]
        rsp = [a.tolist() for a in rsptmp]
        inp = [a.tolist() for a in inptmp]
        pairs = [a.tolist() for a in pairstmp]

        sen.pop(0)
        rsp.pop(0)
        inp.pop(0)
        pairs.pop(0)

        sen = np.asarray(sen)
        rsp = np.asarray(rsp)
        inp = np.asarray(inp)
        pairs = np.asarray(pairs)

        if return_unique:

            _, ids = np.unique(sen, return_index=True)
            sen = sen[np.sort(ids)]
            rsp = rsp[np.sort(ids)]
            pairs = pairs[np.sort(ids)]
            inp = inp[np.sort(ids), :]

        if reshape_back:
            sen = np.hstack(sen)
            rsp = np.hstack(rsp)
            pairs = np.hstack(pairs)
            inp = np.concatenate([np.vstack(a) for a in inp], axis=1)

        return sen, rsp, inp, pairs
