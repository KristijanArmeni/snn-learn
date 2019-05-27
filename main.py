
# OWN MODULES
from snn_params import Params
from snn_network import SNN
from util import save, load
from stimuli import Dataset
import numpy as np

# VIZUALIZATION
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook')

# ===== PARAMETERS ===== #

parameters = Params(tmin=0, tmax=0.05)

# ===== TRAIN DATASET ===== #

dataset = load("/project/3011085.04/snn/data/raw/stimuli_1-4.pkl")

train_ds = Dataset(sequence=dataset[0:int(len(dataset)*0.6)][0],
                   response=dataset[0:int(len(dataset)*0.6)][1],
                   encoding=dataset[0:int(len(dataset)*0.6)][2])

save(train_ds, "/project/3011085.04/snn/data/interim/train_ds.pkl")

# ===== LOAD TUNING PARAMETERS

t100 = load("/project/3011085.04/snn/data/raw/tuning-100")
t500 = load("/project/3011085.04/snn/data/raw/tuning-500")
t950 = load("/project/3011085.04/snn/data/raw/tuning-950")

p = [t100, t500, t950]

N = [100, 500, 950]
Wi = [2.3, 2.3, 2.3]

Wr = []
for i, l in enumerate(p):
    Wr.append(np.round(l["recurrent"][1]*1e9, 4)*1e-9)

# define time windows
tois = [[0, 0.05], [0, 0.01],[0.01, 0.02],[0.02, 0.03], [0.03, 0.04], [0.04, 0.05]]

# ===== 100, 500, 950 NEURONS, RESET STATES, temporal shifting ===== #

for i, n in enumerate(N):

    print("#=====*****=====#")
    print("Running network N={:d}".format(n))

    # preallocate array
    x = np.ndarray(shape=(len(tois), train_ds.sequence.shape[0], n))

    net = SNN(params=parameters, n_neurons=n, input_dim=8, output_dim=2, syn_adapt=True)
    net.config_input_weights(mean=0.4, density=0.50, seed=55)
    net.config_recurrent_weights(density=0.1, ex=0.8, seed=155)

    # apply the selected weight scaling
    net.w["input"] *= 2.3
    net.w["recurrent"] *= Wr[i]

    # Configure recording matrices
    net.config_recording(n_neurons=net.neurons["N"], t=parameters.sim["t"], dataset=train_ds, downsample=None)

    # configure input current
    step = parameters.step_current(t=net.recording["t_orig"], on=0, off=0.05, amp=4.4e-9)

    net.train(dataset=train_ds, current=step, reset_states="sentence")

    # Run averging over selected temporal windows
    for i, toi in enumerate(tois):
        x[i, :, :] = net.avg_states(toi=toi)  # average membrane voltage

    print("Saving output ...")
    save(net, "/project/3011085.04/snn/data/raw/net-{}.pkl".format(net.neurons["N"]))
    save(x, "/project/3011085.04/snn/data/interim/train_x-{}.pkl".format(net.neurons["N"]))

    del net, x
