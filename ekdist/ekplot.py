import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from ekdist import eklib

def plot_stability_intervals(rec, open=True, shut=True, popen=True, window=50):
    opma, shma, poma = eklib.moving_average_open_shut_Popen(rec.opint[:-1], rec.shint, window=window)
    x = np.linspace(0, np.prod(opma.shape), num=np.prod(opma.shape), endpoint=True)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    if open:
        ax.semilogy(x, opma, 'g', label='Open periods')
    if shut:
        ax.semilogy(x, shma, 'r', label='Shut periods')
    if popen:
        ax.semilogy(x, poma, 'b', label='Popen')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
                        borderaxespad=0.)
    ax.set_xlabel('Interval number')
    return fig
    
def plot_stability_amplitudes(rec, window=1):
    all_resolved_opamp = np.array(rec.rampl)[np.where( np.fabs(np.asarray(rec.rampl)) > 0.0)]
    amps = eklib.moving_average(all_resolved_opamp, window)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.plot(amps, '.g')
    #ax.set_ylim([0, 1.2 * max(amps)])
    ax.set_ylabel('Amplitude, pA')
    ax.set_xlabel('Interval number')
    print('Average open amplitude = ', np.average(amps))
    return fig

def plot_fitted_amplitude_histogram(rec, fc, n=2, nbins=20, gauss=True):
    long_opamp = eklib.amplitudes_openings_longer_Tr(rec, fc, n)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.hist(long_opamp, nbins, density=True, alpha=0.6, color='g')
    if gauss:
        mu, std = norm.fit(long_opamp)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        ax.set_title("Fit results: mu = %.2f,  std = %.2f" % (mu, std))
    ax.set_xlim([0, 1.2 * max(long_opamp)])
    ax.set_xlabel('Amplitude, pA')
    ax.set_ylabel('Frequency')
    print('Range of amplitudes: {0:.3f} - {1:.3f}'.
          format(min(long_opamp), max(long_opamp)))
    return fig      