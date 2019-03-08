
import numpy as np


def avg_frate(recording):

    data = np.concatenate(recording["count"])
    stim_time = 0.05  # 50 milliseconds stimulation time
    n_trl = data.shape[0]

    total_time = n_trl*stim_time
    total_spikes = np.sum(data, axis=0)
    frate = total_spikes/total_time

    return frate
