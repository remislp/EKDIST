import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib import ticker

from ekdist import eklib

def stability_intervals(rec, open=True, shut=True, popen=True, window=50):
    opma, shma, poma = eklib.moving_average_open_shut_Popen(
                       rec.periods.get_open_intervals()[:-1], 
                       rec.periods.get_shut_intervals(), 
                       window=window)
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

###############################################################################
# Dwell time histograms: x-log / y-sqrt
def __histogram_bins_per_decade(X):
    nbdec = 12
    if (len(X) <= 300): nbdec = 5 
    if (len(X) > 300) and (len(X) <= 1000): nbdec = 8
    if (len(X) > 1000) and (len(X) <= 3000): nbdec = 10
    return nbdec

def __bin_width(X):
    return math.exp(math.log(10.0) / float(__histogram_bins_per_decade(X)))

def __exponential_scale_factor(X, pars, tres):
    tau, area = eklib._theta_unsqueeze(pars)
    return (len(X) * math.log10(__bin_width(X)) * math.log(10) *
            (1 / np.sum(area * np.exp(-tres / tau))))

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
    """
    dx = __bin_width(X)
    xend = math.exp(math.ceil(math.log(max(X)))) # round up maximum value
    nbin = int(math.log(xend / tres) / math.log(dx)) # number of bins
    my_bins = tres * np.array([dx**i for i in range(nbin+1)])
    hist, bin_edges = np.histogram(X, bins=my_bins)
    xout = [x for pair in zip(bin_edges, bin_edges) for x in pair]
    yout = [0] + [y for pair in zip(hist, hist) for y in pair] + [0]
    return xout, yout

def histogram_xlog_ysqrt_data(X, tres, fitpars=None, tcrit=None, xlabel='Dwell times'):
    """ Plot dwell time histogram in log x and square root y. """
    xout, yout= prepare_xlog_hist(X, tres)
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.semilogx(xout, np.sqrt(yout))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('sqrt(frequency)')
    if fitpars is not None:
        scale = __exponential_scale_factor(X, fitpars, tres)
        t = np.logspace(math.log10(tres), math.log10(2 * max(X)), 512)
        ax.plot(t, np.sqrt(scale * t * eklib.myexp(fitpars, t)), '-b')
        tau, area = eklib._theta_unsqueeze(fitpars)
        for ta, ar in zip(tau, area):
            ax.plot(t, np.sqrt(scale * t * (ar / ta) * np.exp(-t / ta)), '--b')
    if tcrit is not None:
        for tc in np.asarray(tcrit):
            ax.axvline(x=tc, color='g')
