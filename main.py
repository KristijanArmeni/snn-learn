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
import os

# VIZUALIZATION
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

dirs = Paths()

# ===== PARAMETERS ===== #

parameters = Params(tmin=0, tmax=0.05)
# write parameters to csv's
parameters.to_csv(path=os.path.join(dirs.home, "doc", "tables"))

# ===== LOAD THE DATASET ===== #

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

if sys.argv[1] == "demo-neuron":

    # Initialize parameters
    parameters = Params(tmin=0, tmax=0.7, tau_gsra=0.4, tau_gref=0.002)
    # vars(parameters)  # show the contents

    # create a step current
    step = parameters.step_current(t=parameters.sim["t"], on=0.2, off=0.5, amp=2.5e-9)
    step = step[np.newaxis, :]  # make sure step has an extra dimension corresponding to the number of neurons

    # initialize a network consisting of a single neuron
    net = SNN(params=parameters, n_neurons=1, input_dim=1, output_dim=2)

    # configure weight matrices
    # net.config_input_weights(mean=0.4, density=0.50, seed=1)
    net.config_recurrent_weights(density=1, ex=1, seed=2)
    net.w["recurrent"] *= 0

    # configure initial conditions
    states = dict.fromkeys(["V", "I_rec", "gsra", "gref", "spikes"])
    states["V"] = np.zeros(net.neurons["N"],)
    states["I_rec"] = np.zeros(net.neurons["N"],)
    states["gsra"] = np.zeros(net.neurons["N"],)
    states["gref"] = np.zeros(net.neurons["N"],)
    states["spikes"] = np.zeros(net.neurons["N"], dtype=bool)

    states["V"][0] = -0.07  # start from resting membrane potential
    states["I_rec"][0] = 0

    net.gsra["tau"] = 0.04

    V, spikes, count, gsra, gref, I_rec = net.forward(I_in=step, states=states, dt=parameters.sim["dt"])

    # store the variables
    save(V, dirs.raw + "/membrane_demo.pkl")
    save(spikes, dirs.raw + "/spikes_demo.pkl")
    save(I_rec, dirs.raw + "/Irec_demo.pkl")
    save(gref, dirs.raw + "/gref_demo.pkl")
    save(gsra, dirs.raw + "/gsra_demo.pkl")


if sys.argv[1] == "development":

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
        net.config_input_weights(mean=0.4, density=0.50)
        net.config_recurrent_weights(density=0.1, ex=0.8)

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
            x[i, :, :] = net.sample_states(toi=toi)  # average membrane voltage

        print("Saving output ...")
        # save(net, p.raw + "/net-{}.pkl".format(net.neurons["N"]))
        np.save(file=dirs.interim + "/states-{}.npy".format(net.neurons["N"]), arr=x)

        del net, x


# ===== VARY SYNAPTIC TIME CONSTANT ===== #

elif sys.argv[1] == "main-simulation":

    tuning_ds = Dataset(sequence=ds[0:800][0], response=ds[0:800][1], encoding=ds[0:800][2])
    full_ds = Dataset(sequence=ds[2000::][0], response=ds[2000::][1], encoding=ds[2000::][2])

    # initialise variables controling simulation parameters
    N = 1000
    resetting = ["sentence"]
    suffix = []
    tau_values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]  # tau_gsra == 0.001 simply turns off the adaptation parameter
    time_windows = [[0, 0.05], [0, 0.01], [0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05]]

    # initialize parameters
    parameters = Params(tmin=0, tmax=0.05)

    # create current kernel
    step = parameters.step_current(t=parameters.sim["t"], on=0, off=0.05, amp=4e-9)

    # network instance
    net = SNN(params=parameters, n_neurons=N, input_dim=8, output_dim=2, syn_adapt=True)

    for kk in np.arange(0, 10)[9::]:

        # select seeds
        net.w["input_seed"] = parameters.w["input-seeds"][kk]
        net.w["recurrent_seed"] = parameters.w["recurrent-seeds"][kk]

        for k, reset in enumerate(resetting):

            infix = None

            # for constructing output file name
            if reset == "sentence":
                infix = "A"
            elif reset is None:
                infix = "B"

            # preallocate array for storing states
            x = np.ndarray(shape=(len(tau_values), len(time_windows), full_ds.sequence.shape[0], N))

            # variable for storing tuning parameters
            r = {"appInn": np.ndarray(shape=(len(tau_values),)),
                 "appRec": np.ndarray(shape=(len(tau_values),)),
                 "fRate-tune": np.ndarray(shape=(len(tau_values), N)),
                 "fRate": np.ndarray(shape=(len(tau_values), N))}

            for j, tau_gsra in enumerate(tau_values):

                if tau_gsra == 0.0:
                    net.syn_adapt = False  # turn off adaptation term if tau_gsra is set to 0
                else:
                    net.syn_adapt = True   # make sure adaptation term is on otherwise

                # set the gsra and tune the network
                net.gsra["tau"] = tau_gsra

                tune_params_file = dirs.raw + "/tuning_{}-{}-s{:02d}-{}.pkl".format(N, infix, kk+1, tau_gsra)

                # ===== RATE TUNING ===== #
                if os.path.isfile(tune_params_file):

                    print("Loading {}".format(tune_params_file))
                    tune_params = load(tune_params_file)
                    net.w["input_scaling"] = tune_params["input"][1]
                    net.w["recurrent_scaling"] = tune_params["recurrent"][1]

                else:
                    print("\n=====Tuning the network [N = {}, tau = {}, subject {}]=====".format(N, tau_gsra, kk+1))
                    print("Adaptation term == ", net.syn_adapt)

                    # creates values in net.wscale to be used below
                    sel = net.rate_tuning2(parameters=parameters, input_current=step, reset_states=reset, dataset=tuning_ds,
                                           init_scales=[1.8, 4e-9],
                                           targets=[2, 5], increments=(0.2, 0.4e-9), margins=[0.2, 0.5],
                                           warmup=True, warmup_size=0.375,
                                           N_max=40, skip_input=False,
                                           tag="[N = {}, tau = {}, subject {}]".format(N, tau_gsra, kk+1))

                    print(net.w["input_scaling"], net.w["recurrent_scaling"])
                    # store these params for later on
                    r["appInn"][j] = net.w["input_scaling"]
                    r["appRec"][j] = net.w["recurrent_scaling"]
                    r["fRate-tune"][j, :] = net.avg_frate(stim_time=0.05, samples=None)

                # ===== SIMULATION ===== #
                print("\n[s{:02d}] Running network of N = {} with tau_gsra = {:f}".format(kk+1, N, tau_gsra))
                print("Adaptation term == ", net.syn_adapt)

                # define connectivity matrices
                net.config_input_weights(mean=0.4, density=0.50)
                net.config_recurrent_weights(density=0.1, ex=0.8)
                net.config_recording(n_neurons=N, t=parameters.sim["t"], dataset=full_ds, downsample=5)

                # apply global connectivity scaling parameters
                net.w["input"] *= net.w["input_scaling"]
                net.w["recurrent"] *= net.w["recurrent_scaling"]

                # run simulation
                net.train(dataset=full_ds, current=step, reset_states=reset)

                # write firing rate
                r["fRate"][j, :] = net.avg_frate(stim_time=0.05, samples=None)  # average over all samples (default)

                # Run averaging over selected temporal windows
                print("\nCollecting states ...")
                for i, toi in enumerate(time_windows):

                    # take average over whole window if the entire window is specified
                    if toi == [0, 0.05]:
                        print("Mean over toi = ", toi)
                        x[j, i, :, :] = net.sample_states(toi=toi, type="mean")  # average membrane voltage
                    # otherwise record a single sample point
                    else:
                        print("Sampling at t = ", toi[0])
                        x[j, i, :, :] = net.sample_states(toi=toi, type="point")  # sample point membrane voltage

                if not os.path.isfile(tune_params_file):
                    save(sel, tune_params_file)

            print("Saving {}-{}-s{:02d} output to {}...".format(N, infix, kk+1, dirs.interim + '/newtime'))

            # save network parameters
            net.params_to_csv(path=dirs.interim + '/newtime' + "/params_{}-{}-s{:02d}.csv".format(N, infix, kk+1))

            np.save(file=dirs.interim + '/newtime' + "/states_{}-{}-s{:02d}".format(N, infix, kk+1), arr=x)
            save(r, dirs.raw + '/newtime' + "/rates_{}-{}-s{:02d}.pkl".format(N, infix, kk+1))
