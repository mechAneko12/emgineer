
import numpy as np
import matplotlib.pyplot as plt

def plot_spikes(spikes,
                win=None,
                title='title'):
    if win is not None:
        spikes = spikes[win:]
    x_s = (np.arange(spikes.shape[0]) ) /2048
    cmap = ['#de3c3a', '#de9c3a', '#d8de3a', '#70de3a', '#3ade79', '#3adedb', '#3a7bde', '#633ade', '#d33ade', '#de3a86']# plt.get_cmap("tab10")
    i_c = 0
    for i in range(spikes.shape[1]):
        spike_row = spikes[:, i]
        if i_c >= len(cmap):
            i_c = 0
        color = cmap[i_c]
        i_c += 1
        for j in range(spikes.shape[0]):
            if spike_row[j] == 1:
                _x = x_s[j]
                plt.plot([_x, _x], [i+0.1, i+0.9], color=color, lw=1)
    plt.title(title)
    plt.xlabel('time[s]')
    # plt.ylabel('')
    plt.show()
    