import numpy as np
import csv
import os
import pandas as pd


class Params(object):

    def __init__(self, **kwargs):
        """fixme"""

        self.task = {
                "alphabet": np.array(["1", "2", "A", "B", "C", "X", "Y", "Z"]),
                "name": "12AX"
            }

        self.memb = {
            "E": kwargs.get("E", -0.07),
            "V_reset": kwargs.get("V_reset", -0.08),
            "V_thr": kwargs.get("V_thr", -0.05),
            "R": kwargs.get("R", 1e6),
            "tau": kwargs.get("tau", 0.01)
            }

        self.sim = {
            "tmin": kwargs.get("tmin", 0),
            "tmax": kwargs.get("tmax", 0.6),
            "dt": kwargs.get("dt", 0.001),
            "t": (np.arange(start=kwargs.get("tmin", 0)*1000,
                            stop=(kwargs.get("tmax", 0.6)*1000),
                            step=kwargs.get("dt", 0.001)*1000)
                  )/1000
            }

        self.gsra = {
            "tau": kwargs.get("tau_gsra", 0.4),
            "delta": kwargs.get("delta_gsra", 15e-9),
            "E_r": kwargs.get("E_gsra", -0.08)
            }

        self.gref = {
            "tau": kwargs.get("tau_gref", 0.002),
            "delta": kwargs.get("delta_gref", 200e-9),
            "E_r": kwargs.get("E_gref", -0.08)
            }

        self.syn = {
            "tau": kwargs.get("tau_syn", 0.015),
            "delta_g": kwargs.get("delta_syn", 10e-9)
            }

        self.w = {
            "input-seeds": np.random.RandomState(100).choice(np.arange(0, 10000), 10).tolist(),
            "recurrent-seeds": np.random.RandomState(1000).choice(np.arange(0, 10000), 10).tolist(),
                  }

        self.data = dict.fromkeys(["train", "validate", "test"])
        self.inp_cur = dict.fromkeys(["I_amp", "t_on", "t_off", "type"])

    def config_input_current(self, I_amp=4.4e-9, t_on=0.3, t_off=0.35):
        """config_input_current(I_amp, t_on, t_off) configures the
        input current parameters and fills the self.inp_curr dict."""

        self.inp_cur["I_amp"] = I_amp
        self.inp_cur["t_on"] = t_on
        self.inp_cur["t_off"] = t_off

    @staticmethod
    def step_current(t, on, off, amp):
        """

        step_current(t, on, off, amp) creates a step current for the duration
        of t, starting at timepoint on, ending at off with the strength of amp

        :param t:    1d array, time axis defining simulation length
        :param on:   scalar, time point when the current is applied
        :param off:  scalar, time point defining when the current injection is turned off
        :param amp:  scalar, strength of the input current for the stimulation period
        :return: I   1d array, input current time-series
        """
        I = np.zeros((len(t),))              # create empty output array
        onind = (np.abs(t - on)).argmin()    # define onset
        offind = (np.abs(t - off)).argmin()  # find offset

        I[onind:(offind+1)] = amp

        return I

    def split_data(self, data, train=0.8, validate=0.1, test=0.1):

        total_n = len(data)

        self.data['train'] = train
        self.data['validate'] = validate
        self.data['test'] = test

    def to_csv(self, path):

        dicts = {'membrane': self.memb,
                 'synapse': self.syn,
                 'gref': self.gref,
                 'gsra': self.gsra,
                 'sim': self.sim,
                 'connectivity': self.w}

        for dictkey in dicts:

            filename = os.path.join(path, "{}_params.csv".format(dictkey))
            df = None
            if dictkey != 'connectivity':
                df = pd.DataFrame().from_dict(dicts[dictkey], orient='index', columns=['value'])
                df.reset_index(level=0, inplace=True)
                df.rename(columns={'index': 'name'}, inplace=True)
            elif dictkey == 'connectivity':
                df = pd.DataFrame().from_dict(dicts[dictkey])

            df.to_csv(path_or_buf=filename, index=False)

