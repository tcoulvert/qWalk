import os

import matplotlib.pyplot as plt

def plot_time_binned_array(time_binned_array, output_dirpath, filename, density=False):
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    bins = [bin for bin in range(1, len(time_binned_array)+2)]

    fig, ax = plt.subplots()
    ax.hlines(0, xmin=min(bins), xmax=max(bins), colors='k')
    ax.hist(
        bins[:-1],
        bins,
        weights=time_binned_array,
        density=density,
        histtype='step'
    )
    ax.set_xlabel('Time bins')
    ax.set_ylabel('Likelyhood of falling in bin')
    ax.set_title(filename)
    fig.savefig(os.path.join(output_dirpath, filename))
    plt.close(fig)
    
