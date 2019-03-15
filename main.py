
from stimuli import generate_sequence, encode_input, Dataset
from snn_params import Params
from snn_network import SNN

# Initialize parameters
params = Params()
params.config_sim(tmin=0, tmax=0.6, dt=0.001)

seq, rsp, out_seq = generate_sequence(n_inner=4, n_outer=160, seed=95)
inp = encode_input(sequence=seq)

dat = Dataset(sequence=seq, response=rsp, encoding=inp)  # inherit the torch Dataset class properties

# Set up the network architecture and define parameters
net = SNN()

net.init_neurons(n=1000)  # Number of neurons
net.config_membrane(E=-0.07, V_reset=-0.08, V_thr=-0.05, R=1e6, tau=0.01)  # Membrane parameters

# Synaptic parameters
net.config_syn(tau=0.05, delta_g=10e-9)
net.init_weights(dim=[8, 1000, 2])  # define sizes of weight matrices

net.config_input_weights(mean=0.4, density=0.05)
net.config_recurrent_weights(density=0.1, ex=0.8)
net.weight_scaling(input_scaling=1, recurrent_scaling=1.9e-9)

# Spike-rate adaptation and refractoriness

net.config_gsra(tau=0.4, delta=15e-9, E_r=-0.08)
net.config_gref(tau=0.002, delta=200e-9, E_r=-0.08)

# Configure recording matrices

net.config_recording(n_neurons=net.neurons["N"], t=params.sim["T"], dataset=dat, downsample=2)

# Create step current for input
step = params.step_current(t=net.recording["t_orig"],
                                   on=0.3,
                                   off=0.35,
                                   amp=4.4e-9)

net.train(dataset=dat, current=step)


# PLOTS

from matplotlib import pyplot as plt

e = 0

d = net.recording["V"][e]
s = net.recording["spikes"][e]
c = net.recording["count"][e]
x = net.recording["t"][e]


plt.figure()
plt.imshow(d[4, :, :], aspect="auto")
plt.show()

plt.close("all")
plt.plot(x, dat[:, :, 0].T)
plt.show()

plt.imshow(s[:, :, 0], aspect="auto", cmap="gray_r")
