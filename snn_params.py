

import numpy as np

class Params(object):

    def __init__(self):
        """fixme"""

        self.task = dict.fromkeys(['alphabet', 'name'])
        self.task['alphabet'] = np.array(["1", "2", "A", "B", "C", "X", "Y", "Z"])
        self.task['name'] = '12AX'

        self.data = dict.fromkeys(["train", "validate", "test"])
        self.sim = dict.fromkeys(["dt", "tmin", "tmax", "T"])
        self.inp_cur = dict.fromkeys(["I_amp", "t_on", "t_off", "type"])

    def config_sim(self, tmin=0, tmax=0.6, dt=0.001):
        """config_sim(tmin, tmax, dt) configures the
        simulation parameters.
        """
        self.sim["dt"] = dt
        self.sim["tmin"] = tmin
        self.sim["tmax"] = tmax

        # Create integer axis in seconds
        self.sim["T"] = np.arange(start=tmin*1000, stop=(tmax*1000)+1, step=dt*1000)
        self.sim["T"] /= 1000  # convert back to milliseconds

    def config_input_current(self, I_amp=4.4e-9, t_on=0.3, t_off=0.35):
        """config_input_current(I_amp, t_on, t_off) configures the
        input current parameters and fills the self.inp_curr dict."""

        self.inp_cur["I_amp"] = I_amp
        self.inp_cur["t_on"] = t_on
        self.inp_cur["t_off"] = t_off

    @staticmethod
    def step_current(t, on, off, amp):

        I = np.zeros((len(t),))
        onind = np.where(t == on)[0]
        offind = np.where(t == off)[0]

        I[onind[0]:offind[0]] = amp

        return I

    def split_data(self, data, train=0.8, validate=0.1, test=0.1):

        total_n = len(data)

        self.data['train'] = train
        self.data['validate'] = validate
        self.data['test'] = test
