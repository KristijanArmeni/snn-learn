"""Script for tuning average network firing rate

USAGE EXAMPLE:

python rate_tuning.py <[100, 500, 1200]> <300> <code_version> <path_for_saving>

SYNTAX

python rate_tuning.py <network_sizes> <number_trials> <path_for_saving>
"""

# OWN MODULES

from snn_params import Params
from snn_network import SNN
from util import save, load
from stimuli import Dataset
import sys


sizes = list(map(int, sys.argv[1].strip('[]').split(',')))
n_trl = int(sys.argv[2])
version = sys.argv[3]

# ===== PARAMETERS ===== #

parameters = Params(tmin=0, tmax=0.05)

# ===== TRAIN DATASET ===== #

dat = load("/project/3011085.04/snn/data/raw/stimuli_1-4.pkl")

# ===== RATE TUNING: NO STATE RESETTING ===== #

# subset the dataset
tmp = Dataset(sequence=dat[0:n_trl][0], response=dat[0:n_trl][1], encoding=dat[0:n_trl][2])

# seeds for N = [100, 500]: in: 55, rec:155

for N in sizes:

    print("Tuning network with N = {}".format(N))

    net = SNN(params=parameters, n_neurons=N, input_dim=8, output_dim=2, syn_adapt=True)

    # Configure recording matrices
    net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=tmp, downsample=False)
    step = parameters.step_current(t=net.recording["t_orig"], on=0, off=0.05, amp=4.4e-9)

    if version == "rate_tuning":
        sel = net.rate_tuning(parameters=parameters, input_current=step, dataset=tmp,
                              input_scale=1.4, input_step=0.1,
                              rec_scale=1e-9, rec_step=0.2e-9,
                              targets=[1.8, 2.2, 4.5, 5.5],
                              N_max=25, skip_input=False)

    elif version == "rate_tuning2":

        sel = net.rate_tuning2(parameters=parameters, input_current=step, dataset=tmp,
                               init_scales=[1.4, 1e-9],
                               targets=[2, 5], margins=[0.2, 0.5],
                               N_max=25, skip_input=False)

    save(sel, sys.argv[4] + "/" + "tuning-{}".format(N))
