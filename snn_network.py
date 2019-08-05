
import numpy as np
import pandas as pd
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
                  "input_seed": None,
                  "recurrent_seed": None,
                  "input_scaling": None,
                  "recurrent_scaling": None}

        # parameters
        self.sim = params.sim
        self.memb = params.memb  # membrane parameters
        self.syn = params.syn  # synaptic parameters
        self.gsra = params.gsra  # spike-rate adaptation
        self.gref = params.gref  # refractory conductance

        # for storing output states
        self.recording = dict.fromkeys(["V", "t_orig", "t", "gsra", "gref", "spikes", "labels", "downsample"])

    def __call__(self, I_in, states, dt):

        V, spikes, spike_count, gsra, gref, I_rec = self.forward(self, I_in, states, dt)

        return V, spikes, spike_count, gsra, gref, I_rec

    def config_input_weights(self, mean=0.4, density=0.05):

        sel = np.random.RandomState(seed=self.w["input_seed"]).random_sample(size=self.w['input'].shape) > density  # define connection probability

        self.w["input"][:] = np.random.RandomState(seed=self.w["input_seed"]).exponential(scale=mean, size=self.w['input'].shape)
        self.w["input"][sel] = 0  # put 95% of connections to 0

        return self

    def config_recurrent_weights(self, density=0.1, ex=0.8):

        self.w["recurrent"][:] = np.random.RandomState(seed=self.w["recurrent_seed"]).random_sample(size=self.w["recurrent"].shape)
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
        self.recording["spikes"] = np.zeros((len(dataset.sequence), n_neurons, len(t_orig)), dtype=bool)
        self.recording["count"] = np.zeros((len(dataset.sequence), n_neurons))
        self.recording["axes"] = ["neuron_nr", "time", "trial_nr"]
        self.recording["downsample"] = downsample

    def forward(self, I_in, states, dt):
        """
        .forward()

        :param I_in:
        :param states:
        :param dt:
        :return: V
        :return:
        """

        # Initialize local versions of variables
        samples = len(self.sim["t"])     # simulate with original high sample time axis
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
            alpha_gsra = np.exp(-dt / tau_gsra)
            alpha_gref = np.exp(-dt / tau_gref)
        else:
            c = 0
            alpha_gsra = 0
            alpha_gref = 0

        # Stimulation loop (start indexing from 1)
        for k in np.arange(1, samples):

            # Reset voltage for neurons that spiked
            if np.any(fired):
                V[fired, k-1] = self.memb["V_reset"]

            # gsra and gref decay
            gsra[:, k] = gsra[:, k-1] * alpha_gsra
            gref[:, k] = gref[:, k-1] * alpha_gref

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

        return V, spikes, spike_count, gsra, gref, I_rec

    def train(self, dataset, current, reset_states="sentence"):
        """

        :param dataset: subclass of pyTorch Dataset class, needs to have fields .sequence, .encoding, .response
        :param current: 1d array, input current applied on every trial
        :param reset_states: dict, containing initial conditions for simulation on each trial
        :return:
        """

        dt = self.recording["t_orig"][1]-self.recording["t_orig"][0]  # infer simulation time step

        # create dict for storing dynamic variables
        states = dict.fromkeys(["V", "I_rec", "gsra", "gref", "spikes"])

        states["V"] = np.zeros(self.neurons["N"],)
        states["I_rec"] = np.zeros(self.neurons["N"],)
        states["gsra"] = np.zeros(self.neurons["N"],)
        states["gref"] = np.zeros(self.neurons["N"],)
        states["spikes"] = np.zeros(self.neurons["N"], dtype=bool)

        # Loop over trials
        for i in tqdm(range(len(dataset.sequence))):
            # print('Trial {:d}'.format(i))

            # project the stimulus onto the network
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

            # store the states of dynamic variables and take them to the next stimulation
            states["V"] = V[:, -1]
            states["I_rec"] = I_rec[:, -1]
            states["gsra"] = gsra[:, -1]
            states["gref"] = gref[:, -1]
            states["spikes"] = spikes[:, -1]

            # Downsample the recording if needed
            if self.recording["downsample"]:

                # dwonsample dynamic variables
                self.recording["V"][i, :, :] = V[:, ::self.recording["downsample"]]
                self.recording["gsra"][i, :, :] = gsra[:, ::self.recording["downsample"]]
                self.recording["gref"][i, :, :] = gref[:, ::self.recording["downsample"]]

            else:
                self.recording["V"][i, :, :] = V
                self.recording["gsra"][i, :, :] = gsra
                self.recording["gref"][i, :, :] = gref

            # store total recordings for output
            self.recording["spikes"][i, :, :] = spikes
            self.recording["count"][i, :] = count

    def avg_frate(self, stim_time=0.05, samples=None):

        if samples is None:
            n_trl = self.recording["count"].shape[0]
            samples = np.arange(0, n_trl)
        else:
            n_trl = self.recording["count"][samples, :].shape[0]

        print("\nComputing firing rate over {} samples, from sample nr. {} to {}".format(n_trl, samples[0], samples[-1]))

        total_time = n_trl * stim_time                          # total time (in sec)
        total_spikes = np.sum(self.recording["count"][samples, :], axis=0)  # total number of spikes

        firing_rate = total_spikes / total_time  # firing rate per neuron

        print("Firing rate of {} Hz over window {:.1f} seconds ({:d} samples)".format(np.mean(firing_rate), total_time, n_trl))

        return firing_rate

    def sample_states(self, toi=None, type=None):

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

        if type == "mean":
            # take average over selected time period
            out = np.nanmean(a=self.recording["V"][:, :, ton:toff], axis=2)
        elif type == "point":
            # take a single sample point
            out = self.recording["V"][:, :, ton]

        return out

    def rate_tuning(self, parameters, input_current, dataset, input_scale, input_step, rec_scale, rec_step, targets,
                    N_max=25, skip_input=False):

        params = ["input", "recurrent"]
        sel = {params[0]: [None, None, None, None],
               params[1]: [None, None, None, None],
               }

        if skip_input:
            params = ["recurrent"]
            sel["input"][1] = input_scale

        for target_weights in params:

            sel[target_weights][0] = self.neurons["N"]  # log network size

            f1 = 0
            c = 1

            scales = []
            rates = []
            bisect = False

            # create weight matrices
            self.config_input_weights(mean=0.4, density=0.50)
            self.config_recurrent_weights(density=0.1, ex=0.8)

            # disconnect internal connectivity if tuning input
            if target_weights == "input":
                self.w["recurrent"][:] *= 0
                a = input_scale
                s = input_step
                x1 = targets[0]
                x2 = targets[1]
            elif target_weights == "recurrent":
                self.w["input"][:] *= sel["input"][1]  # scale input by the selected value
                a = rec_scale
                s = rec_step
                x1 = targets[2]
                x2 = targets[3]

            while (f1 < x1 or f1 > x2) and c < N_max:

                print("\n====*****=====")
                print("[Iteration {:d} (N={:d})]".format(c, self.neurons["N"]))
                print("Initial f. rate:", round(f1, 3))
                print("Tuning {} weights with".format(target_weights), a, "scale")

                # scale the input weights
                self.w[target_weights][:] *= a

                self.config_recording(n_neurons=self.neurons["N"], t=parameters.sim["t"], dataset=dataset, downsample=False)
                self.train(dataset=dataset, current=input_current, reset_states="sentence")

                f0 = f1
                f1 = np.mean(self.avg_frate())

                rates.append(f1)
                scales.append(a)

                # unscale the weights back
                self.w[target_weights][:] *= (1 / a)

                print("New average f. rate:", round(f1, 3))

                # check if we skipped the target range, then bisect
                if c > 1 and not bisect and (f0 < x1 and f1 > x2):
                    print("Too steep increase. Bisecting.", a, "and", b)
                    a0 = a
                    a = (a + b) / 2  # bisect the value
                    bisect = True

                elif c > 1 and not bisect and (f0 > x2 and f1 < x1):
                    print("Too steep descent. Bisecting.", a, "and", b)
                    a0 = a
                    a = (a + b) / 2
                    bisect = True

                # bisection results in f(a) to fall below the target range
                elif bisect and f1 < x1 and f0 > x2:
                    print("Bisection below the range. Bisecting (a0, a).", a0, "and", a)
                    a = (a0 + a) / 2  # take the old value as the upper point
                    bisect = True

                    # bisection results in f(a) to fall below the target range
                elif bisect and f0 > x2 and f1 > x2:
                    print("Bisection above the range. Bisecting.", a0, "and", a)
                    a = (a0 + a) / 2  # take the old value as the upper point
                    bisect = True

                elif f0 < x1 and f1 < x2:
                    print("Rates too low. Increasing 'a' by.", s)
                    b = a   #
                    a += s  # increase the scale if below target range
                    bisect = False

                elif f0 > x1 and f1 > x2:
                    print("Rates too high. Decreasing 'a' by.", s)
                    b = a
                    a -= s  # decrease the scale if above target range
                    bisect = False

                c += 1

            if c >= N_max:
                print("{} tuning did not converge".format(target_weights))
            else:
                print("{} tuning converged for {} scale".format(target_weights, target_weights), a, "and rate", f1)
                sel[target_weights][1] = a

            sel[target_weights][2] = scales
            sel[target_weights][3] = rates

        return sel

    def rate_tuning2(self, parameters=None, input_current=None, reset_states="sentence", dataset=None,
                     init_scales=None, targets=None, margins=None, skip_input=False,
                     warmup=False, warmup_size=None, N_max=25, tag=None):

        params = ["input", "recurrent"]
        sel = {params[0]: [None, None, None, None],
               params[1]: [None, None, None, None],
               }

        if skip_input:
            params = ["recurrent"]
            # fill in the output dict
            sel["input"][0] = self.neurons["N"]
            sel["input"][1] = init_scales[0]
            sel["input"][2] = "skipped_tuning"
            sel["input"][3] = "skipped_tuning"

        for i, target_weights in enumerate(params):

            sel[target_weights][0] = self.neurons["N"]  # log network size

            spikeRate = None
            targetRate = targets[i]
            margin = margins[i]
            success = False
            abort = False
            c = 0
            x1 = None
            x2 = None
            bisect = False

            resApp = None
            scales = []
            rates = []

            # create weight anew prior to tuning
            self.config_input_weights(mean=0.4, density=0.5)
            self.config_recurrent_weights(density=0.1, ex=0.8)

            # disconnect internal connectivity if tuning input weights
            if target_weights == "input":
                self.w["recurrent"][:] *= 0
                resApp = init_scales[i]      # starting value for tuning
                # s = input_step

            elif target_weights == "recurrent":
                self.w["input"][:] *= sel["input"][1]  # scale input by the selected value
                resApp = init_scales[i]
                # s = rec_step

            while not success and not abort:

                c += 1  # increase counter

                if c > N_max:
                    print("{} tuning did not converge.".format(target_weights))
                    self.w["{}_scaling".format(target_weights)] = resApp
                    sel[target_weights][1] = np.nan
                    sel[target_weights][2] = scales
                    sel[target_weights][3] = rates
                    return sel

                print("\n====*****=====")
                print(tag)
                print("[Iteration {:d} (N={:d}), warmup = {}, warmup_size={:d} %]".format(c, self.neurons["N"],
                                                                                        warmup, int(warmup_size*100)))
                print("Initial f. rate:", spikeRate)
                if target_weights == "recurrent":
                    print("Selected input scale: {:.4e}".format(sel["input"][1]))
                print("Tuning {} weights with {:.4e} scale".format(target_weights, resApp))

                # scale the input weights
                self.w[target_weights][:] *= resApp

                self.config_recording(n_neurons=self.neurons["N"], t=parameters.sim["t"], dataset=dataset, downsample=False)
                self.train(dataset=dataset, current=input_current, reset_states=reset_states)

                if warmup:  # only measure avg rate from a sample point later in simulation
                    begin_spl = int(round(dataset.sequence.shape[0]*warmup_size))  # start averging at this sample point
                    end_spl = dataset.sequence.shape[0]               # average until the final sample
                else:
                    begin_spl = 0
                    end_spl = dataset.sequence.shape[0]

                # spikeRate = self.
                spikeRate = np.mean(self.avg_frate(stim_time=0.05, samples=np.arange(begin_spl, end_spl)))

                rates.append(spikeRate)
                scales.append(resApp)

                # unscale the weights back
                self.w[target_weights][:] *= (1 / resApp)

                # Determine the starting points (x1, x2) for bisection\

                if (targetRate-margin) < spikeRate < (targetRate+margin):

                    x1 = resApp
                    x2 = resApp

                    print("{} tuning done. Scaling parameter:".format(target_weights), resApp)
                    print("Average spike rate good! Spike rate:", spikeRate)
                    sel[target_weights][1] = resApp
                    sel[target_weights][2] = scales
                    sel[target_weights][3] = rates
                    success = True

                if not bisect and not success and not abort:

                    if spikeRate < targetRate:    # we're undershooting, store as x1 and increase resApp

                        print("Spike rate {:f} Hz below targetRate {:f} Hz. ".format(spikeRate, targetRate))

                        x1 = resApp

                        # positive increments
                        if x2 is None and target_weights == "recurrent":
                            resApp += 0.2e-9
                        elif x2 is None and target_weights == "input":
                            resApp += 0.2

                    else:                         # we're overshooting, store as x2 and increase resApp

                        print("Spike rate {:f} Hz above targetRate {:f} Hz. ".format(spikeRate, targetRate))

                        x2 = resApp

                        if x1 is None:
                            resApp *= 0.7

                    if (x1 and x2) is not None:

                        bisect = True
                        print("Found initial values for bisection:")
                        print("x1 = {:.4e} | x2 = {:.4e}".format(x1, x2))

                if bisect and not success and not abort:

                    if spikeRate < (targetRate-margin):

                        x1 = resApp

                    elif spikeRate > (targetRate+margin):

                        x2 = resApp

                    # check if conditions are good
                    if (targetRate-margin) < spikeRate < (targetRate+margin):

                        print("Bisection for {} weights converged!".format(target_weights))
                        print("Spike rate:", spikeRate)
                        print("{} weight:".format(target_weights), resApp)
                        sel[target_weights][1] = resApp
                        sel[target_weights][2] = scales
                        sel[target_weights][3] = rates
                        success = True

                    else:

                        print("Bisecting x1 = {:.4e} and x2 = {:.4e}".format(x1, x2))
                        resApp = (x1 + x2)/2

            self.w["{}_scaling".format(target_weights)] = resApp
            sel[target_weights][1] = resApp
            sel[target_weights][2] = scales
            sel[target_weights][3] = rates

        return sel

    def params_to_csv(self, path=None):

        dat = [self.memb, self.w, self.syn, self.gsra, self.gref]

        print("Writing parameters to {}".format(path))
        pd.DataFrame(data=dat).to_csv(path_or_buf=path, index=False)
