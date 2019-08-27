"""Module containing layer definitions"""

# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedding(nn.Embedding):

    def __init__(self, vocabulary=None, dim=None, **kwargs):
        super(Embedding, self).__init__(num_embeddings=len(vocabulary.id2word), embedding_dim=dim)

        self.vocabulary = vocabulary


class FC(nn.Module):

    def __init__(self, n_in, n_out, nonlinearity=None):
        super(FC, self).__init__()

        self.W = nn.Linear(in_features=n_in, out_features=n_out, bias=True)
        self.nonlin = nonlinearity

    def forward(self, inputs, **kwargs):

        if self.nonlin is not None:
            return self.nonlin(self.W(inputs))
        else:
            return self.W(inputs)


class SRN(nn.Module):

    def __init__(self, n_in, n_out, bs=None):
        super(SRN, self).__init__()

        self.W = nn.Linear(in_features=n_in, out_features=n_out, bias=True)
        self.U = nn.Linear(in_features=n_out, out_features=n_out, bias=True)

        self.h = torch.zeros((bs, n_out))

    def forward(self, inputs, hidden):

        # RNN forward (activations and hidden)
        a = self.W(inputs) + self.U(hidden)  # activations
        self.h = torch.tanh(a)               # New hidden state

        return self.h


class GRU(nn.Module):
    """
    Gated recurrent unit layer.

    """
    def __init__(self, n_in, n_out, bs=None):
        super(GRU, self).__init__()

        # Input weights
        self.W_z = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # update gate
        self.W_r = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # reset gate
        self.W_n = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # new gate

        # Recurrent weights
        self.U_z = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # update
        self.U_r = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # reset
        self.U_n = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # new gate

        # States and gates
        self.h = torch.zeros((bs, n_out), requires_grad=False)
        self.r = torch.zeros((bs, n_out), requires_grad=False)
        self.z = torch.zeros((bs, n_out), requires_grad=False)

    def forward(self, inputs=None, hidden=None):

        r = torch.sigmoid(self.W_r(inputs) + self.U_r(hidden))  # reset gate
        z = torch.sigmoid(self.W_z(inputs) + self.U_z(hidden))  # update gate

        n = torch.tanh(self.W_n(inputs) + (self.W_n(inputs) + self.U_n(r*hidden)))  # new gate

        self.h = z*hidden + (1 - z)*n  # new hidden state

        return self.h, (r, z)

    def count_params(self):

        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class LSTM(nn.Module):

    def __init__(self, n_in, n_out, bs=None):
        super(LSTM, self).__init__()

        # Input weights
        self.W_i = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # input (remember) gate
        self.W_f = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # forget gate
        self.W_o = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # output gate
        self.W_c = nn.Linear(in_features=n_in, out_features=n_out, bias=True)  # new memory

        # Recurrent weights
        self.U_i = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # update
        self.U_f = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # reset
        self.U_o = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # output
        self.U_c = nn.Linear(in_features=n_out, out_features=n_out, bias=True)  # new memory

        # States
        self.h = torch.zeros((bs, n_out), requires_grad=False)
        self.c = torch.zeros((bs, n_out), requires_grad=False)

    def forward(self, inputs=None, hidden=None, memory=None):

        i = torch.sigmoid(self.W_i(inputs) + self.U_i(hidden))    # input gate
        f = torch.sigmoid(self.W_f(inputs) + self.U_f(hidden))    # forget gate
        o = torch.sigmoid(self.W_o(inputs) + self.U_o(hidden))    # output gate

        c_ = torch.tanh(self.W_c(inputs) + self.U_c(hidden))      # candidate memory state

        self.c = f*memory + i*c_       # new memory state
        self.h = o*torch.tanh(self.c)  # new hidden state

        return self.h, self.c, (i, f, o)


class AttentionHead(nn.Module):

    def _init_(self):
        super(AttentionHead, self).__init__()

        self.c = None  # context vector
        self.K = None  # keys
        self.q = None  # query

    def forward(self, inputs, context):

        pass

    def compatibility(self, keys, query):
        """
        energy function or alignement model
        :param keys:
        :param query:
        :return:
        """
        e = None

        return e

    def attention(self, context, annotation):

        e = None

        return e