"""
main.py "train/test"
"""
# OWN MODULES
from snn_params import Params
from snn_network import SNN
from stimuli import Dataset
from util import save, load, Paths
import numpy as np
import sys

# VIZUALIZATION
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

dirs = Paths()

# ===== PARAMETERS ===== #

parameters = Params(tmin=0, tmax=0.05)

# ===== TRAIN DATASET ===== #

print("Loading stimuli...")
ds = load(dirs.raw + "/12ax-12k.pkl")

set1_idx = np.arange(0, 2000)
set2_idx = np.arange(0, len(ds.sequence))

# ===== LOAD TUNING PARAMETERS

t100 = load(dirs.raw + "/tuning-100")
t500 = load(dirs.raw + "/tuning-500")
t1000 = load(dirs.raw + "/tuning-1000")

p = [t100, t500, t1000]
N = [100, 500, 1000]

Wi = []
for i, l in enumerate(p):
    Wi.append(l["input"][1])      # read input scaling parameter

Wr = []
for i, l in enumerate(p):
    Wr.append(l["recurrent"][1])

if sys.argv[1] == "exploratory":

    # define time windows
    tois = [[0, 0.05], [0, 0.01], [0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05]]

    # ===== 100, 500, 950 NEURONS, RESET STATES, temporal shifting ===== #

    for i, n in enumerate(N):

        print("\n#=====*****=====#")
        print("Running network N={:d}".format(n))

        # use larger dataset for N == 1000
        if n in [100, 500]:
            dataset = Dataset(sequence=ds[set1_idx][0], response=ds[set1_idx][1], encoding=ds[set1_idx][2])
        elif n == 1000:
            dataset = Dataset(sequence=ds[set2_idx][0], response=ds[set2_idx][1], encoding=ds[set2_idx][2])

        # preallocate array
        print(dataset.sequence.shape)
        x = np.ndarray(shape=(len(tois), dataset.sequence.shape[0], n))

        net = SNN(params=parameters, n_neurons=n, input_dim=8, output_dim=2, syn_adapt=True)
        net.config_input_weights(mean=0.4, density=0.50, seed=55)
        net.config_recurrent_weights(density=0.1, ex=0.8, seed=155)

        # apply the selected weight scaling
        net.w["input"] *= Wi[i]
        net.w["recurrent"] *= Wr[i]

        # Configure recording matrices
        net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=dataset, downsample=5)

        # configure input current
        step = parameters.step_current(t=net.recording["t_orig"], on=0, off=0.05, amp=4.4e-9)

        net.train(dataset=dataset, current=step, reset_states="sentence")
        print(net.avg_frate())
        print("Mean:", np.mean(net.avg_frate()))

        # Run averging over selected temporal windows
        for i, toi in enumerate(tois):
            x[i, :, :] = net.avg_states(toi=toi)  # average membrane voltage

        print("Saving output ...")
        # save(net, p.raw + "/net-{}.pkl".format(net.neurons["N"]))
        np.save(file=dirs.interim + "/states-{}.npy".format(net.neurons["N"]), arr=x)

        del net, x


# ===== VARY SYNAPTIC TIME CONSTANT ===== #

elif sys.argv[1] == "gsra-effect":

    tuning_ds = Dataset(sequence=ds[0:500][0], response=ds[0:500][1], encoding=ds[0:500][2])
    full_ds = Dataset(sequence=ds[2000::][0], response=ds[2000::][1], encoding=ds[2000::][2])

    resetting = ["sentence", None]
    suffix = []
    values = [0.05, 0.075, 0.1, 0.15, 0.2, 0.4, 0.5, 0.7, 1.0, 1.5]
    N = 1000
    time_windows = [[0, 0.05], [0, 0.01], [0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05]]

    # parameters
    parameters = Params(tmin=0, tmax=0.05)
    step = parameters.step_current(t=parameters.sim["t"], on=0, off=0.05, amp=4.4e-9)

    net = SNN(params=parameters, n_neurons=1000, input_dim=8, output_dim=2, syn_adapt=True)

    # preallocate array
    x = np.ndarray(shape=(len(values), len(time_windows), full_ds.sequence.shape[0], N))

    for k, reset in enumerate(resetting):

        for j, tau_gsra in enumerate(values):

            print("\n=====Tuning the network [N = {}, tau = {}]=====".format(N, tau_gsra))

            # set the gsra and tune the network
            net.gsra["tau"] = tau_gsra

            # creates values in net.wscale to be used below
            net.rate_tuning2(parameters=parameters, input_current=step, reset_states=reset, dataset=tuning_ds,
                             init_scales=[1.4, 1e-9],
                             targets=[2, 5], margins=[0.2, 0.5],
                             N_max=25, skip_input=False)

            print(net.wscale["input"], net.wscale["recurrent"])

            print("[{:d}] Running network of N = {} with tau_gsra = {:f}".format(k, N, tau_gsra))

            net.config_input_weights(mean=0.4, density=0.50, seed=55)
            net.config_recurrent_weights(density=0.1, ex=0.8, seed=155)
            net.config_recording(n_neurons=N, t=parameters.sim["t"], dataset=full_ds, downsample=5)

            net.w["input"] *= net.wscale["input"]
            net.w["recurrent"] *= net.wscale["recurrent"]

            net.train(dataset=full_ds, current=step, reset_states=reset)

            print("Mean firing rate:", np.mean(net.avg_frate()))

            # Run averaging over selected temporal windows
            print("Averaging ...")
            for i, toi in enumerate(time_windows):
                x[j, i, :, :] = net.avg_states(toi=toi)  # average membrane voltage

        print("Saving output ...")
        if reset == "sentence":
            suffix = "reset"
        elif reset is None:
            suffix = "noreset"

        np.save(file=dirs.interim + "/states-{}_{}_tau".format(N, suffix), arr=x)
