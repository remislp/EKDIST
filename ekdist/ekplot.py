import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib import ticker

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

###############################################################################
# Dwell time histograms: x-log / y-sqrt
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
    xend = math.exp(math.ceil(math.log(max(X)))) # round up maximum value
    nbin = int(math.log(xend / tres) / math.log(dx)) # number of bins
    my_bins = tres * np.array([dx**i for i in range(nbin+1)])
    hist, bin_edges = np.histogram(X, bins=my_bins)
    xout = [x for pair in zip(bin_edges, bin_edges) for x in pair]
    yout = [0] + [y for pair in zip(hist, hist) for y in pair] + [0]
    return xout, yout, dx

def histogram_xlog_ysqrt_data(X, tres, xlabel='Dwell times'):
    """
    Plot dwell time histogram in log x and square root y.
    """
    xout, yout, dx = prepare_xlog_hist(X, tres)
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.semilogx(xout, yout)
    mscale.register_scale(SquareRootScale)
    ax.set_yscale('sqrtscale')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('sqrt(frequency)')

###############################################################################
# TODO: Consider moving this class somewhere else.
class SquareRootScale(mscale.ScaleBase):
    """ Class for generating square root scaled axis for probability density
    function plots.
    https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python
    """
    name = 'sqrtscale'
    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
    def get_transform(self):
        """ Set the actual transform for the axis coordinates. """
        return self.SqrTransform()
    def set_default_locators_and_formatters(self, axis):
        """ Set the locators and formatters to reasonable defaults. """
        axis.set_major_formatter(ticker.ScalarFormatter())

    class SqrTransform(mtransforms.Transform):
        input_dims, output_dims = 1, 1
        is_separable = True
        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            """ Take numpy array and return transformed copy. """
            return np.sqrt(a)
        def inverted(self):
            """ Get inverse transform. """
            return SquareRootScale.InvertedSqrTransform()

    class InvertedSqrTransform(mtransforms.Transform):
        input_dims, output_dims = 1, 1
        is_separable = True
        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            return np.power(a, 2)
        def inverted(self):
            return SquareRootScale.SqrTransform()
