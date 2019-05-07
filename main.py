
# OWN MODULES
from stimuli import generate_sequence, encode_input, Dataset
from snn_params import Params
from snn_network import SNN
from util import load, save
from plots import plot_scores, plot_trial, plot_time

# DATA MANIPULATION
import pandas as pd
import numpy as np

# VIZUALIZATION
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')
import seaborn as sns
#from cycler import cycler
#default_cycler = cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
#plt.rc('axes', prop_cycle=default_cycler)

# ===== PARAMETERS ===== #

parameters = Params(tmin=0, tmax=0.05)

# ===== DATASET ===== #

seq, rsp, out_seq = generate_sequence(n_inner=4, n_outer=170, seed=95)
inp = encode_input(sequence=seq)
dat = Dataset(sequence=seq, response=rsp, encoding=inp)  # inherit the torch Dataset class properties


# ===== NETWORK SETUP===== #

net = SNN(params=parameters, n_neurons=1000, input_dim=8, output_dim=2)
net.config_input_weights(mean=0.4, density=0.50, seed=1)
net.config_recurrent_weights(density=0.1, ex=0.8, seed=2)

# ===== INPUT CURRENT ===== #

# Configure recording matrices
net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=dat, downsample=None)

step = parameters.step_current(t=net.recording["t_orig"], on=0, off=0.05, amp=4.4e-9)


# ===== RATE TUNING ===== #

# subset the dataset
tmp = Dataset(sequence=dat[0:50][0], response=dat[0:50][1], encoding=dat[0:50][2])

# disconnect internal synapses
input_scales = [1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4]

# disconnect the recurrent the weights to zero
net.w["recurrent"][:] = 0

frate = []

for scale in input_scales:

    # scale the input weights
    net.w["input"][:] *= scale

    net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=tmp, downsample=None)
    net.train(dataset=tmp, current=step)

    frate.append(net.avg_frate())

    # unscale the weights back
    net.w["input"][:] *= (1/scale)

# organize input into a pd dataframe

df = pd.DataFrame(data=np.array(frate).T, columns=np.array(input_scales).astype(str))
df = pd.melt(df, value_vars=np.array(input_scales).astype(str), var_name="scale", value_name="rate")

df.to_csv(path_or_buf="/project/3011085.04/snn/data/interim/tuning_inputw.csv")
df.to_pickle(path="/project/3011085.04/snn/data/interim/tuning_inputw.pkl")

df1 = pd.read_csv("/project/3011085.04/snn/data/interim/tuning_inputw.csv", )

# make some plots

plt.figure()
plt.plot(df["scale"], df["rate"], 'o')
plt.title("Input weight tuning (N = 1000)")
plt.xlabel("Input weight (a.u.)")
plt.ylabel("firing rate (Hz)")
plt.show()

plt.close()

# Tune recurrent synapses

recurrent_scales = [1e-10, 1e-9, 3e-9, 3.5e-9, 3.7e-9, 4e-9, 5e-9, 7e-9]

# reconfigure the weights
net.config_input_weights(mean=0.4, density=0.5, seed=1)
net.config_recurrent_weights(density=0.1, ex=0.8, seed=2)
net.w["input"] *= 2.1  # scale input weights

frate = []
std = []
for scale in recurrent_scales:

    # keep input weights the same
    net.w["recurrent"][:] *= scale

    net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=tmp, downsample=None)
    net.train(dataset=tmp, current=step)

    frate.append(net.avg_frate())

    # unscale the recurrent weights
    net.w["recurrent"][:] *= (1/scale)


df = pd.DataFrame(data=np.array(frate).T, columns=np.array(recurrent_scales).astype(str))
df = pd.melt(df, value_vars=np.array(recurrent_scales).astype(str), var_name="scale", value_name="rate")

df.to_csv(path_or_buf="/project/3011085.04/snn/data/interim/tuning_recurrentw.csv")
df.to_pickle(path="/project/3011085.04/snn/data/interim/tuning_recurrentw.pkl")

# make some plots

plt.figure()
plt.plot(df["scale"], df["rate"], 'o')
plt.title("Input weight tuning (N = 1000)")
plt.xlabel("Input weight (a.u.)")
plt.ylabel("firing rate (Hz)")
plt.show()

sns.catplot(data=df, x="scale", y="rate", color="black")

# ===== STIMULATE THE NETWORK ===== #

net = SNN(params=parameters, n_neurons=1000, input_dim=8, output_dim=2)
net.config_input_weights(mean=0.4, density=0.50, seed=1)
net.config_recurrent_weights(density=0.1, ex=0.8, seed=2)

# apply the selected weight scaling
net.w["input"] *= 2.1
net.w["recurrent"] *= 3.7e-9

# Configure recording matrices
net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=dat, downsample=None)
step = parameters.step_current(t=net.recording["t_orig"], on=0, off=0.05, amp=4.4e-9)

net.train(dataset=dat, current=step)

# ===== AVERAGE STATES ===== #

# Define the feature and target arrays
x = net.avg_states()  # average membrane voltage
y = rsp               # response (0 or 1)


# ===== SAVE OUTPUT ===== #

save(net, "/project/3011085.04/snn/data/raw/recording.pkl")
save(x, "/project/3011085.04/snn/data/interim/average_states.pkl")
save(y, "/project/3011085.04/snn/data/interim/responses.pkl")