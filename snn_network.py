
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm


class SNN(object):

    def __init__(self, params, n_neurons=1000, input_dim=None, output_dim=None, syn_adapt=True):

        # store number of neurons
        self.neurons = {"N": n_neurons}
        self.syn_adapt = syn_adapt

        # configure weight matrices
        self.w = {"input": np.ndarray(shape=(n_neurons, input_dim)),
                  "recurrent": np.ndarray(shape=(n_neurons, n_neurons)),
                  "output": np.ndarray(shape=(n_neurons, output_dim)),
                  "input_scaling": 0,
                  "recurrent_scaling": 0}

        # parameters
        self.task = params.task
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

        return self

    def config_recurrent_weights(self, density=0.1, ex=0.8, seed=None):

        self.w["recurrent"][:] = np.random.RandomState(seed=seed).random_sample(size=self.w["recurrent"].shape)
        sel = self.w["recurrent"] > density
        self.w["recurrent"][sel] = 0  # set selected weights to 0

        # set excitatory and inhibitory connections
        n_exc = round(self.w["recurrent"].shape[0]*ex)  # number of exc. synapses
        self.w["recurrent"][:, n_exc:] *= -5  # make inhibitory connections five times stronger

        return self

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

    def forward(self, I_in, states, dt):

        # Initialize local versions of variables
        samples = len(self.recording["t_orig"])     # simulate with original high sample time axis
        n = self.neurons["N"]                       # number of neurons

        I_rec = np.zeros((n, samples))              # recurrent input
        V = np.zeros((n, samples))                  # membrane potential
        V[:, :] = self.memb["E"]                    # initialize with E

        gsra = np.zeros((n, samples))               # sra conductance
        gref = np.zeros((n, samples))               # refractory conductance
        spikes = np.zeros((n, samples), dtype=bool) # spike log
        fired = np.zeros((n,), dtype=bool)          # logical for active neurons
        theta = self.memb["V_thr"]                  # threshold variable

        tau = self.memb["tau"]
        tau_syn = self.syn["tau"]
        E = self.memb["E"]
        R = self.memb["R"]
        E_gsra = self.gsra["E_r"]
        tau_gsra = self.gsra["tau"]
        tau_gref = self.gref["tau"]

        # assign the past states to the first samples of dynamic variables
        V[:, 0] = states["V"][:]
        gsra[:, 0] = states["gsra"][:]
        gref[:, 0] = states["gref"][:]
        I_rec[:, 0] = states["I_rec"][:]
        fired[:] = states["spikes"][:]

        # set the parameter which controls the adaptation term in the update eq
        if self.syn_adapt:
            c = 1
        else:
            c = 0

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
                c * ((V[:, k-1] - E_gsra) * R * (gsra[:, k-1] + gref[:, k-1]))  # adaptation term

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

        return V, spikes, spike_count, gsra, gref, I_rec

    def train(self, dataset, current, reset_states="sentence"):

        dt = self.recording["t"][1]-self.recording["t"][0]  # infer time step

        # create dict for storing dynamic variables
        states = dict.fromkeys(["V", "I_rec", "gsra", "gref"])

        states["V"] = np.zeros(self.neurons["N"],)
        states["I_rec"] = np.zeros(self.neurons["N"],)
        states["gsra"] = np.zeros(self.neurons["N"],)
        states["gref"] = np.zeros(self.neurons["N"],)
        states["spikes"] = np.zeros(self.neurons["N"], dtype=bool)

        # Loop over trials
        for i in tqdm(range(len(dataset.sequence))):
            # print('Trial {:d}'.format(i))

            stimulated_neurons = self.w["input"] @ dataset.encoding[:, i]
            I_in = np.outer(stimulated_neurons, current)

            # keep states for non onset symbols, else they're reset
            if reset_states == "sentence":

                if dataset.sequence[i] == "1" or dataset.sequence[i] == "2":

                    # reset the states of dynamic variables and take them to the next stimulation
                    states["V"][:] = self.memb["E"]  # reset to resting potential
                    states["I_rec"][:] = 0           # zero currents
                    states["gsra"][:] = 0
                    states["gref"][:] = 0
                    states["spikes"][:] = False      # clear firing history

            V, spikes, count, gsra, gref, I_rec = self.forward(I_in=I_in, states=states, dt=dt)

            # store the states of dynamic variable and take them to the next stimulation
            states["V"] = V[:, -1]
            states["I_rec"] = I_rec[:, -1]
            states["gsra"] = gsra[:, -1]
            states["gref"] = gref[:, -1]
            states["spikes"] = spikes[:, -1]

            # store total recordings for output
            self.recording["V"][i, :, :] = V
            self.recording["spikes"][i, :, :] = spikes
            self.recording["count"][i, :] = count
            self.recording["gsra"][i, :, :] = gsra
            self.recording["gref"][i, :, :] = gref

    def avg_frate(self):

        stim_time = 0.05  # in seconds, 50 milliseconds stimulation time
        n_trl = self.recording["count"].shape[0]

        total_time = n_trl * stim_time
        total_spikes = np.sum(self.recording["count"], axis=0)

        frate = total_spikes / total_time  # firing rate per neuron

        # mean over active neurons
        #activated = self.recording["spikes"].any(axis=2).any(axis=0)
        mean = np.mean(frate)
        std = np.std(frate)

        return frate

    def avg_states(self, toi=None):

        """
        Compute average membrane voltage per neuron per trial

        :param data: array of membrane voltage (trial x neurons x time) created in net.config_recording().
        :param toi: list of integers, onset and offset times
        :return out: ndarray (trials x neurons), average voltage
        """

        if toi is None:
            toi = [0, 0.05]

        # get timing sample points
        ton = (np.abs(self.recording["t"] - toi[0])).argmin()
        toff = (np.abs(self.recording["t"] - toi[1])).argmin()

        # take average over selected time period
        out = np.nanmean(a=self.recording["V"][:, :, ton:toff], axis=2)

        return out

    def rate_tuning(self, parameters, input_current, dataset, input_scale, input_step, rec_scale, rec_step, targets,
                    skip_input=False):

        if not skip_input:

            fr = 0
            c = 1

            input_scales = []
            input_rates = []
            not_converge = False

            # disconnect internal connectivity
            self.w["recurrent"][:] *= 0

            while fr < targets[0] or fr > targets[1]:

                print("[Iteration {:d} (N={:d})]".format(c, self.neurons["N"]))
                print("Average f. rate: {:f} Hz".format(fr))
                print("Tuning input weights for {:f}...".format(input_scale))

                # scale the input weights
                self.w["input"][:] *= input_scale

                self.config_recording(n_neurons=self.neurons["N"], t=parameters.sim["t"], dataset=dataset, downsample=False)
                self.train(dataset=dataset, current=input_current, reset_states="sentence")

                fr_old = fr
                fr = np.mean(self.avg_frate())

                input_rates.append(fr)
                input_scales.append(input_scale)

                # unscale the weights back
                self.w["input"][:] *= (1 / input_scale)

                # check if we skipped the taret range
                if c > 1 and fr_old < targets[0] and fr > targets[1]:
                    print("Too steep increase. Can't settle in the range. Stopping.")
                    not_converge = True
                    break
                elif c > 1 and fr_old > targets[1] and fr < targets[0]:
                    print("Too steep descent. Can't settle in the range. Stopping.")
                    not_converge = True
                    break

                c += 1

                # Modify input scale
                if fr < targets[0]:
                    input_scale += input_step   # increase the scale if below target range
                elif fr > targets[1]:
                    input_scale -= input_step    # decrease the scale if above target range

            if not_converge:
                print("Input tuning did not converge")
                return

        self.config_input_weights(mean=0.4, density=0.50, seed=1)
        self.config_recurrent_weights(density=0.1, ex=0.8, seed=2)

        c = 1
        fr = 0
        rec_scales = []
        rec_rates = []
        not_converge = False

        # use the final value of input_scale
        self.w["input"][:] *= input_scale

        while fr < targets[2] or fr > targets[3]:

            print("Iteration {:d}".format(c))
            print("Current average f. rate for N={}: {:f}...".format(self.neurons["N"], fr))
            print("Tuning recurrent weights for {:f} and input scale {:f}...".format(rec_scale, input_scale))
            print(rec_scale)

            # scale the input weights
            self.w["recurrent"][:] *= rec_scale

            self.config_recording(n_neurons=self.neurons["N"], t=parameters.sim["t"], dataset=dataset, downsample=False)
            self.train(dataset=dataset, current=input_current, reset_states="sentence")

            fr_old = fr
            fr = np.mean(self.avg_frate())

            rec_rates.append(fr)
            rec_scales.append(rec_scale)

            # unscale the weights back
            self.w["recurrent"][:] *= (1 / rec_scale)

            if fr < targets[2]:
                rec_scale += rec_step  # increase the scale if below target range
            elif fr > targets[3]:
                rec_scale -= rec_step  # decrease the scale if above target range

            # check if we skipped the taret range
            if c > 1 and fr_old < targets[2] and fr > targets[3]:
                print("Too steep increase. Can't settle in the range. Stopping.")
                not_converge = True
                break
            elif c > 1 and fr_old > targets[3] and fr < targets[2]:
                print("Too steep descent. Can't settle in the range. Stopping.")
                not_converge = True
                break

            c += 1

        if not_converge:
            print("Recurrent tuning did not converge")
            return

        self.w["input_scaling"] = input_scale
        self.w["recurrent_scaling"] = rec_scale

        return input_scale, rec_scale, [input_rates, input_scales], [rec_rates, rec_scales]