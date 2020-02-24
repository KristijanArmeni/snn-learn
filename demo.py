
import numpy as np

# visualization
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from stimuli import generate_sequence, encode_input, Dataset
from snn_params import Params
from snn_network import SNN
from util import load, save, Paths
from plots import plot_trial

dirs = Paths()

# ===== DEMO: MEMBRANE DYNAMICS ===== #

# Initialize parameters
parameters = Params(tmin=0, tmax=0.7, tau_gsra=0.4, tau_gref=0.002)

# create a step current
step = parameters.step_current(t=parameters.sim["t"], on=0.2, off=0.5, amp=2.5e-9)
step = step[np.newaxis, :]  # make sure step has an extra dimension corresponding to the number of neurons

# initialize network instance
net0 = SNN(params=parameters, n_neurons=1, input_dim=1, output_dim=2, syn_adapt=False)
net = SNN(params=parameters, n_neurons=1, input_dim=1, output_dim=2)

# configure weight matrices
net0.config_recurrent_weights(density=1, ex=1)
net0.w["recurrent"] *= 0
net.config_recurrent_weights(density=1, ex=1)
net.w["recurrent"] *= 0

states = dict.fromkeys(["V", "I_rec", "gsra", "gref", "spikes"])
states["V"] = np.zeros(net.neurons["N"],)
states["I_rec"] = np.zeros(net.neurons["N"],)
states["gsra"] = np.zeros(net.neurons["N"],)
states["gref"] = np.zeros(net.neurons["N"],)
states["spikes"] = np.zeros(net.neurons["N"], dtype=bool)

states["V"][0] = -0.07  # start from resting membrane potential
states["I_rec"][0] = 0

# run a single trial
V0, spikes0, count0, gsra0, gref0, I_rec0 = net0.forward(I_in=step, states=states, dt=parameters.sim["dt"])

# make tau_gsra 400 msec
net.gsra["tau"] = 0.3
# run a single trial
V, spikes, count, gsra, gref, I_rec = net.forward(I_in=step, states=states, dt=parameters.sim["dt"])

net.gsra["tau"] = 0.1
V2, spikes2, count2, gsra2, gref2, I_rec2 = net.forward(I_in=step, states=states, dt=parameters.sim["dt"])

# plot spikes
V_plot0 = np.copy(V0)*1e3  # conver to mV
V_plot0[:, spikes0[0, :]] = 0.02*1e3

gsra0[:, :] = 0
gref0[:, :] = 0

V_plot = np.copy(V)*1e3  # convert to mV
V_plot[:, spikes[0, :]] = 0.02*1e3  # make spikes more obvious
V_plot2 = np.copy(V2)*1e3  # convert to mV
V_plot2[:, spikes2[0, :]] = 0.02*1e3

# draw the figures

plt.style.use("seaborn-talk")

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 6), sharex='col')
ax[0, 0].plot(net.sim["t"], V_plot0[0, :])
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].tick_params(axis='both', which='major', labelsize=14)
ax[0, 0].tick_params(axis='both', which='major', labelsize=16)


ax[0, 1].plot(net.sim["t"], V_plot2[0, :], label="tau = {:f} msec".format(net.gsra["tau"]*1000))
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].get_yaxis().set_visible(False)
ax[0, 2].plot(net.sim["t"], V_plot[0, :], label="tau = {:f} msec".format(net.gsra["tau"]*1000))
ax[0, 2].get_xaxis().set_visible(False)
ax[0, 2].get_yaxis().set_visible(False)

ax[1, 1].plot(net.sim["t"], gsra2[0, :]*1e9, label="gsra")
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].get_yaxis().set_visible(False)

ax[1, 1].plot(net.sim["t"], gref2[0, :]*1e9, label="gref")
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].get_yaxis().set_visible(False)
ax[1, 1].legend(loc="best")

ax[1, 0].plot(net.sim["t"], gsra0[0, :]*1e9, label="gsra")
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].tick_params(axis='both', which='major', labelsize=14)

ax[1, 0].plot(net.sim["t"], gref0[0, :]*1e9, label="gref")
ax[1, 0].get_xaxis().set_visible(False)
#ax[1, 0].set_ylabel("conductance(S)")
ax[1, 0].set_ylim(ax[1, 1].get_ylim())
ax[1, 0].legend(loc="best")

ax[1, 2].plot(net.sim["t"], gsra[0, :]*1e9, label="gsra")
ax[1, 2].get_xaxis().set_visible(False)
ax[1, 2].get_yaxis().set_visible(False)

ax[1, 2].plot(net.sim["t"], gref[0, :]*1e9, label="gref")
ax[1, 2].get_xaxis().set_visible(False)
ax[1, 2].get_yaxis().set_visible(False)
ax[1, 2].set_ylim(ax[1, 1].get_ylim())
ax[1, 2].legend(loc="best")

ax[2, 0].plot(net.sim["t"], step[0, :]*1e9)
ax[2, 0].set_xlabel("time (sec)")
ax[2, 0].tick_params(axis='both', which='major', labelsize=14)
ax[2, 1].plot(net.sim["t"], step[0, :]*1e9)
ax[2, 1].set_xlabel("time (sec)")
ax[2, 1].get_yaxis().set_visible(False)
ax[2, 1].tick_params(axis='both', which='major', labelsize=14)

ax[2, 2].plot(net.sim["t"], step[0, :]*1e9)
ax[2, 2].set_xlabel("time (sec)")
ax[2, 2].get_yaxis().set_visible(False)
ax[2, 2].tick_params(axis='both', which='major', labelsize=14)
ax[2, 0].set_ylim(-0.2, 4)
ax[2, 1].set_ylim(-0.2, 4)
ax[2, 2].set_ylim(-0.2, 4)
#ax[2, 0].set_ylabel("input current (A)")

ax[0, 0].tick_params(axis='both', which='major', labelsize=14)

plt.savefig(dirs.figures + "/adaptation.svg")
plt.savefig(dirs.figures + "/adaptation.png")
plt.savefig(dirs.figures + "/adaptation.pdf", bbox_inches='tight')


# ===== DEMO: NETWORK DYNAMICS ===== #

ds = load(dirs.raw + "/12ax-12k.pkl")
demo_ds = Dataset(sequence=ds[0:500][0], response=ds[0:500][1], encoding=ds[0:500][2])

# initialise variables controling simulation parameters
N = 1000

# initialize parameters
parameters = Params(tmin=0, tmax=0.05)

# create current kernel
step = parameters.step_current(t=parameters.sim["t"], on=0, off=0.05, amp=4e-9)

# network instance
net = SNN(params=parameters, n_neurons=1000, input_dim=8, output_dim=2, syn_adapt=True)

# load subject-specific tuning and assign it
tune_params = load(dirs.raw + "/tuning_1000-A-s01-0.4.pkl")
net.w["input_scaling"] = tune_params["input"][1]
net.w["recurrent_scaling"] = tune_params["recurrent"][1]

# use connectivity seed for subject 1
net.w["input_seed"] = parameters.w["input-seeds"][0]
net.w["recurrent_seed"] = parameters.w["recurrent-seeds"][0]

# define connectivity
net.config_input_weights(mean=0.4, density=0.50)
net.config_recurrent_weights(density=0.1, ex=0.8)

# scale the weights
net.w["input"] *= net.w["input_scaling"]
net.w["recurrent"] *= net.w["recurrent_scaling"]

# PLOT: CONNECTIVITY
w1 = net.w["input"]

plt.style.use("seaborn")
plt.figure(figsize=(5, 8))
plt.imshow(w1, aspect="auto")
plt.xlabel("input symbol")
plt.xticks(ticks=np.arange(0, 8), labels=parameters.task["alphabet"])
plt.ylabel("receiving neuron")
plt.title("Input connectivity")
c = plt.colorbar()
c.set_label("connection strength")
plt.show()

plt.savefig(dirs.figures + "/input_w.eps")
plt.savefig(dirs.figures + "/input_w.png")

# plot input connectivity matrix
w2 = net.w["recurrent"]

plt.figure(figsize=(8, 8))
plt.style.use("default")
plt.imshow(w2, aspect="auto", cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=-np.max(np.abs(w2)), vmax=np.max(np.abs(w2))))
plt.xlabel("sending neuron")
plt.ylabel("receiving neuron")
plt.title("Internal (recurrent) connectivity")
c = plt.colorbar()
c.set_label("connection strength")
plt.show()

plt.savefig(dirs.figures + "/recurrent_w.eps")
plt.savefig(dirs.figures + "/recurrent_w.png")

# PLOT: DYNAMICS
net.config_recording(n_neurons=1000, t=parameters.sim["t"], dataset=demo_ds, downsample=None)

net.train(dataset=demo_ds, current=step, reset_states="sentence")

dat0 = np.hstack(net.recording["V"][0:20, :, :])
dat1 = np.hstack(net.recording["spikes"][0:20, :, :])

fr = np.sum(dat1, axis=1)
maxid = np.where(fr == np.max(fr))[0][0]
minid = np.where(fr == 0)[0][50]
avgid = np.where(fr == 5)[0][50]

dat0[minid, np.where(dat0[minid, :] == -0.08)[0]-1] = -0.01
dat0[maxid, np.where(dat0[maxid, :] == -0.08)[0]-1] = -0.01
dat0[avgid, np.where(dat0[avgid, :] == -0.08)[0]-1] = -0.01

onsets = np.arange(0, dat0.shape[1], 50)

# create data structure for scatterplot
yv = np.zeros((dat1.shape))
yv[:] = np.nan
for i in range(dat1.shape[0]):
    yv[i, np.where(dat1[i, :])[0]] = i


plt.style.use("seaborn-talk")
fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 1, 1, 3]}, sharex='col', figsize=(16, 7))

t = np.arange(0, 1000)

# plot membrane potentials
ax[0].plot(dat0[minid, :]*1e3, label='non-firing neuron', color='black', alpha=0.7)
ax[0].set_ylim(ax[0].get_ylim()[0], -0.01*1e3)
ax[1].plot(dat0[maxid, :]*1e3, label='{} Hz neuron'.format(np.max(fr)), color='black', alpha=0.7)
ax[2].plot(dat0[avgid, :]*1e3, label='5 Hz neuron', color='black', alpha=0.7)
ax[0].legend(loc="best")
ax[1].legend(loc="best")
ax[2].legend(loc="best")
#ax[0].set_ylabel("membrane potential (mV)")
#ax[0].set_title("Example membrane dynamics")

# plot spikes
#plt.style.use("seaborn")

# use scatterplot

#ax[1].imshow(dat1, aspect="auto")
for i in range(dat1.shape[0]):
    ax[3].scatter(x=t, y=yv[i, :], color='gray', marker='|')
ax[3].vlines(onsets, ymin=0, ymax=50, linestyles="dashed", color='red', linewidth=1.5, label="sentence onset")
ax[3].set_ylim(0, 1000)

for i in range(onsets.shape[0]):
    ax[3].text(x=onsets[i]+2, y=50, s=demo_ds.sequence[i], fontsize=14, color='red')

ax[0].tick_params(axis='y', which='major', labelsize=14)
ax[1].tick_params(axis='y', which='major', labelsize=14)
ax[2].tick_params(axis='y', which='major', labelsize=14)
ax[3].tick_params(axis='y', which='major', labelsize=14)
ax[3].tick_params(axis='x', which='major', labelsize=14)

plt.xlim(0, 1000)

plt.savefig(dirs.figures + "/example-activity.pdf", bbox_inches='tight')
plt.savefig(dirs.figures + "/example-activity.svg")
plt.savefig(dirs.figures + "/example-activity.png")
