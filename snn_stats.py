
import numpy as np


def avg_frate(recording):

    data = np.concatenate(recording["count"])
    stim_time = 0.05  # 50 milliseconds stimulation time
    n_trl = data.shape[0]

    total_time = n_trl*stim_time
    total_spikes = np.sum(data, axis=0)
    frate = total_spikes/total_time

    return frate


def avg_states(data, toi=None):

    """
    Compute average membrane voltage per neuron per trial

    :param data: array of membrane voltage (trial x neurons x time) created in net.config_recording().
    :param toi: list of integers, onset and offset times
    :return out: ndarray (trials x neurons), average voltage
    """

    # get timing sample points
    ton = (np.abs(data-toi[0])).argmin()
    toff = (np.abs(data-toi[1])).argmin()

    # take average over selected time period
    out = np.nanmean(a=data[:, :, ton:toff], axis=2)

    return out

