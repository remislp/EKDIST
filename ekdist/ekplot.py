import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from ekdist import eklib

def stability_intervals(rec, open=True, shut=True, popen=True, window=50):
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
    
def stability_amplitudes(rec, window=1):
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

def histogram_fitted_amplitudes(rec, fc, n=2, nbins=20, gauss=True):
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

def __histogram_bins_per_decade(X):
    nbdec = 12
    if (len(X) <= 300): nbdec = 5 
    if (len(X) > 300) and (len(X) <= 1000): nbdec = 8
    if (len(X) > 1000) and (len(X) <= 3000): nbdec = 10
    return nbdec

def prepare_xlog_hist(X, tres):
    """ Prepare x-log histogram.     

    Parameters
    ----------
    X :  1-D array or sequence of scalar
    tres : float
        Temporal resolution, shortest resolvable time interval. It is
        histogram's starting point.

    Returns
    -------
    xout, yout :  list of scalar
        x and y values to plot histogram.
    dx : float
        Histogram bin width.
    """

    dx = math.exp(math.log(10.0) / float(__histogram_bins_per_decade(X))) # bin width
    xstart = tres    # histogramm starts at
    xend = math.exp(math.ceil(math.log(max(X)))) # round up maximum value
    nbin = int(math.log(xend / xstart) / math.log(dx))
    xaxis = tres * np.array([dx**i for i in range(nbin+1)])

    # Sorts data into bins.
    freq = np.zeros(nbin)
    for i in range(len(X)):
        for j in range(nbin):
            if X[i] >= xaxis[j] and X[i] < xaxis[j+1]:
                freq[j] = freq[j] + 1

    xout = np.zeros((nbin + 1) * 2)
    yout = np.zeros((nbin + 1) * 2)

    xout[0] = xaxis[0]
    yout[0] = 0
    for i in range(0, nbin):
        xout[2*i+1] = xaxis[i]
        xout[2*i+2] = xaxis[i+1]
        yout[2*i+1] = freq[i]
        yout[2*i+2] = freq[i]
    xout[-1] = xaxis[-1]
    yout[-1] = 0

    return xout, yout, dx



def histogram_xlog_ysqrt_data(X, tres):
    """
    Plot dwell time histogram in log x and square root y.
    """
    xout, yout, dx = prepare_xlog_hist(X, tres)
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.semilogx(xout, np.sqrt(yout))
    ax.set_xlabel('Apparent periods')
    ax.set_ylabel('sqrt(frequency)')
