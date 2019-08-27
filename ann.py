
import numpy as np

# pyTorch libraries
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, layers, device, tag) -> None:
        super(Network, self).__init__()

        self.n_hidden = len(layers)-2
        self.hidden_dim = layers[-1].W.weight.shape[1]
        self.layers = nn.Sequential(*layers)
        self.device = device
        self.tag = tag + "-{}-{}".format(self.n_hidden, self.hidden_dim)

    def forward(self, inputs):

        for i, layer in enumerate(self.layers):

            name = type(layer).__name__

            # distinguish between hidden and in/output layers
            if name == "Embedding":

                #print("emb inputs: {}".format(inputs))
                inputs = layer(inputs)  # output is input for next layer

            elif name == "SRN":

                inputs = layer(inputs=inputs, hidden=self.layers[i].h)
                self.layers[i].h = inputs  # write new states for next time step

            elif name == "GRU":

                inputs, _ = layer(inputs=inputs, hidden=self.layers[i].h)
                self.layers[i].h = inputs  # write new states for next time step

            elif name == "LSTM":

                inputs, memory, _ = layer(inputs=inputs, hidden=self.layers[i].h, memory=self.layers[i].c)
                self.layers[i].h = inputs  # write new states for next time step
                self.layers[i].c = memory  # write new memory for next time step

            else:  # it's a fully-connected linear layer

                inputs = layer(inputs)

        output = inputs  # return the result of the last transformation

        return output

    def detach_states(self):

        for i, layer in enumerate(self.layers):

            name = type(layer).__name__

            if name in ["SRN", "GRU"]:

                self.layers[i].h.detach_()

            elif name == "LSTM":

                self.layers[i].h.detach_()
                self.layers[i].c.detach_()

    def init_hidden(self, shape=None):

        for i, layer in enumerate(self.layers):

            name = type(layer).__name__

            if name in ["SRN", "GRU"]:

                self.layers[i].h = torch.zeros(shape).to(self.device)

            elif name == "LSTM":

                self.layers[i].h = torch.zeros(self.layers[i].h.shape).to(self.device)
                self.layers[i].c = torch.zeros(self.layers[i].c.shape).to(self.device)
