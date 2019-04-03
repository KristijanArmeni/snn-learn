
import numpy as np


class SNN(object):

    def __init__(self, params, n_neurons=1000, input_dim=None, output_dim=None):

        # store number of neurons
        self.neurons = {"N": n_neurons}

        # configure weight matrices
        self.w = {"input": np.ndarray(shape=(n_neurons, input_dim)),
                  "recurrent": np.ndarray(shape=(n_neurons, n_neurons)),
                  "output": np.ndarray(shape=(n_neurons, output_dim))
                  }

        # parameters
        self.memb = params.memb  # membrane parameters
        self.syn = params.syn  # synaptic parameters
        self.gsra = params.gsra  # spike-rate adaptation
        self.gref = params.gref  # refractory conductance

        # for storing output states
        self.recording = dict.fromkeys(["V", "t_orig", "t", "gsra", "gref", "spikes", "labels", "downsample"])

    def __call__(self, I_in, dt):

        V, spikes, spike_count, gsra, gref = self.forward(self, I_in=I_in, dt=dt)

        return V, spikes, spike_count, gsra, gref

    def config_input_weights(self, mean=0.4, density=0.05, seed=None):

        sel = np.random.RandomState(seed=seed).random_sample(size=self.w['input'].shape) > density  # define connection probability

        self.w["input"][:] = np.random.RandomState(seed=seed).exponential(scale=mean, size=self.w['input'].shape)
        self.w["input"][sel] = 0  # put 95% of connections to 0

    def config_recurrent_weights(self, density=0.1, ex=0.8, seed=None):

        self.w["recurrent"][:] = np.random.RandomState(seed=seed).random_sample(size=self.w["recurrent"].shape)
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
            gref = gref[:, ::self.recording["downsample"]]

        return V, spikes, spike_count, gsra, gref

    def train(self, dataset, current):

        dt = self.recording["t"][1]-self.recording["t"][0]  # infer time step

        # Loop over trials
        for i in range(len(dataset.sequence)):
            print('Trial {:d}'.format(i))

            stimulated_neurons = self.w["input"] @ dataset.encoding[:, i]
            I_in = np.outer(stimulated_neurons, current)

            V, spikes, count, gsra, gref = self.forward(I_in=I_in, dt=dt)

            self.recording["V"][i, :, :] = V
            self.recording["spikes"][i, :, :] = spikes
            self.recording["count"][i, :] = count
            self.recording["gsra"][i, :, :] = gsra
            self.recording["gref"][i, :, :] = gref
