
import numpy as np
import scipy as sp
from scipy import signal
from scipy import signal
from tqdm import tnrange, tqdm


class SNN(object):

    def __init__(self):

        self.neurons = dict.fromkeys(["N"])
        self.syn = dict.fromkeys(["tau", "delta_g"])  # synaptic parameters
        self.w = dict.fromkeys(["input", "recurrent", "output"])  # weights
        self.memb = dict.fromkeys(["E", "V_reset", "V_thr", "R", "tau"])  # membrane parameters
        self.recording = dict.fromkeys(["V", "t_orig", "t", "gsra", "gref", "spikes", "labels", "downsample"])  # output states
        self.gsra = dict.fromkeys(["tau", "delta", "E_r"])  # spike-rate adaptation
        self.gref = dict.fromkeys(["tau", "delta", "E_r"])  # refractory conductance

    def __call__(self, I_in, dt):

        V, spikes, spike_count, gsra, gref = self.forward(self, I_in=I_in, dt=dt)

        return V, spikes, spike_count, gsra, gref

    def init_neurons(self, n=1000):

        self.neurons["N"] = n

    def init_weights(self, dim):

        self.w['input'] = np.ndarray(shape=(dim[1], dim[0]))
        self.w['recurrent'] = np.ndarray(shape=(dim[1], dim[1]))
        self.w['output'] = np.ndarray(shape=(dim[1], dim[2]))

    def config_membrane(self, E=-0.07, V_reset=-0.08, V_thr=-0.05, R=1e6, tau=0.010):
        """config_membrane() sets the values of membrane parameters in self.memb"""

        self.memb["E"] = E
        self.memb["V_reset"] = V_reset
        self.memb["V_thr"] = V_thr
        self.memb["R"] = R
        self.memb["tau"] = tau

    def config_gsra(self, tau=0.4, delta=15e-9, E_r=-0.08):

        self.gsra["tau"] = tau
        self.gsra["delta"] = delta
        self.gsra["E_r"] = E_r

    def config_gref(self, tau=0.002, delta=200e-9, E_r=-0.08):

        self.gref["tau"] = tau
        self.gref["delta"] = delta
        self.gref["E_r"] = E_r

    def config_syn(self, tau=0.05, delta_g=10e-9):

        self.syn["tau"] = tau
        self.syn["delta_g"] = delta_g

    def config_input_weights(self, mean=0.4, density=0.05, seed=None):

        if seed is not None:
            np.random.seed(seed)  # set fixed seed for debugging
        sel = np.random.random(size=self.w['input'].shape) > density  # define connection probability

        if seed is not None:
            np.random.seed(seed)
        self.w["input"][:] = np.random.exponential(scale=mean, size=self.w['input'].shape)
        self.w["input"][sel] = 0  # put 95% of connections to 0

    def config_recurrent_weights(self, density=0.1, ex=0.8, seed=None):

        if seed is not None:
            np.random.seed(seed)  # set fixed seed for debugging

        self.w["recurrent"][:] = np.random.random(size=self.w["recurrent"].shape)
        sel = self.w["recurrent"] > density
        self.w["recurrent"][sel] = 0  # set selected weights to 0

        # set excitatory and inhibitory connections
        n_exc = round(self.w["recurrent"].shape[0]*ex)  # number of exc. synapses
        self.w["recurrent"][:, n_exc:] *= -5  # make inhibitory connections five times stronger

    def weight_scaling(self, input_scaling=5, recurrent_scaling=1.9e-9):

        self.w["input"][:] *= input_scaling
        self.w["recurrent"][:] *= recurrent_scaling

    def config_recording(self, n_neurons=1000, t=np.linspace(0, 1, 1000), dataset=None, downsample=False):
        """config_recording() creates dict 'self.recording' where network output states are recorded.

        USE AS
        net.config_recording(n_neurons=100, t=np.linspace(0,1,1000), dataset=Dataset, donsample=None)

        INPUTS
        n_neurons       = int, number of neurons in the network
        dataset         = object, pyTorch Dataset class (see Dataset())
        t               = 1d array, time axis for simulation
        downsample      = int, downsampling factor indicating a step size for downsampling t[::downsample]
        """
        t_orig = t

        if downsample is not False:
            t = t[::downsample]  # take evenly spaced samples from the original timeaxis

        self.recording["V"] = np.zeros((len(dataset.sequence), n_neurons, len(t)))
        self.recording["gsra"] = np.zeros((len(dataset.sequence), n_neurons, len(t)))
        self.recording["gref"] = np.zeros((len(dataset.sequence), n_neurons, len(t)))
        self.recording["t_orig"] = t_orig
        self.recording['t'] = t
        self.recording["spikes"] = np.zeros((len(dataset.sequence), n_neurons, len(t)), dtype=bool)
        self.recording["count"] = np.zeros((len(dataset.sequence), n_neurons))
        self.recording["axes"] = ["neuron_nr", "time", "trial_nr"]
        self.recording["downsample"] = downsample

    def forward(self, I_in, dt):

        # Initialize local versions of variables
        samples = len(self.recording["t_orig"])     # simulate with original high sample time axis
        n = self.neurons["N"]                       # number of neurons

        I_rec = np.zeros((n, samples))              # recurrent input
        V = np.zeros((n, samples))                  # membrane potential
        V[:, :] = self.memb["E"]                    # initialize with E

        gsra = np.zeros((n, samples))               # sra conductance
        gref = np.zeros((n, samples))               # refractory conductance
        spikes = np.zeros((n, samples), dtype=bool)  # spike log
        fired = np.zeros((n, samples), dtype=bool)  # logical for active neurons
        theta = self.memb["V_thr"]                  # threshold variable

        tau = self.memb["tau"]
        tau_syn = self.syn["tau"]
        E = self.memb["E"]
        R = self.memb["R"]
        E_gsra = self.gsra["E_r"]
        tau_gsra = self.gsra["tau"]
        tau_gref = self.gref["tau"]

        # Stimulation loop (start indexing from 1)
        for k in np.arange(1, samples):

            # Reset voltage for neurons that spiked
            if np.any(fired):
                V[fired, k-1] = self.memb["V_reset"]

            # gsra and gref decay
            gsra[:, k] = gsra[:, k-1] * (1 - (dt/tau_gsra))
            gref[:, k] = gref[:, k-1] * (1 - (dt/tau_gref))

            # Integrate voltage change
            dV = dt/tau * (E - V[:, k-1]) + \
                R * (I_in[:, k] + I_rec[:, k-1]) - \
                ((V[:, k-1] - E_gsra) * R * (gsra[:, k-1] + gref[:, k-1]))

            # update voltage
            V[:, k] = V[:, k-1] + dV

            # Log spikes
            fired = V[:, k] > theta
            spikes[:, k] = fired

            # Generate recurrent input
            I_rec[:, k] = I_rec[:, k-1] + (self.w["recurrent"] @ spikes[:, k]) - dt * I_rec[:, k-1] * (1/tau_syn)

            # Increment sra and ref conductances
            gsra[fired, k] = gsra[fired, k-1] + self.gsra["delta"]
            gref[fired, k] = gref[fired, k-1] + self.gref["delta"]

        spike_count = np.sum(spikes, axis=1)

        # Downsample the recording if needed
        if self.recording["downsample"] is not False:

            V = V[:, ::self.recording["downsample"]]
            spikes = spikes[:, ::self.recording["downsample"]]
            gsra = gsra[:, ::self.recording["downsample"]]
            gref = gsra[:, ::self.recording["downsample"]]

        return V, spikes, spike_count, gsra, gref

    def train(self, dataset, current):

        dt = self.recording["t"][1]-self.recording["t"][0]  # infer time step

        # Loop over epochs
        for j in tqdm(range(len(dataset.sequence))):
            print('Epoch {:d}'.format(j))

            encodings = dataset.encoding[j]

            # Loop over trials
            for i in range(len(encodings)):
                print('Trial {:d}'.format(i))

                stimulated_neurons = self.w["input"] @ encodings[i, :]
                I_in = np.outer(stimulated_neurons, current)

                V, spikes, count, gsra, gref = self.forward(I_in=I_in, dt=dt)

                self.recording["V"][j][i, :, :] = V
                self.recording["spikes"][j][i, :, :] = spikes
                self.recording["count"][j][i, :] = count
                self.recording["gsra"][j][i, :, :] = gsra
                self.recording["gref"][j][i, :, :] = gsra
