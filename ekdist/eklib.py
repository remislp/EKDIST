import math
import numpy as np
from scipy.optimize import minimize

def moving_average(x, n):
    """ Compute an n period moving average. """
    x = np.asarray(x)
    weights = np.ones(n)
    weights /= weights.sum()
    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def moving_average_open_shut_Popen(opints, shints, window=50):
    """window : moving average interval. """
    opma = moving_average(opints, window)[window-1:] # Moving average for open periods
    shma = moving_average(shints, window)[window-1:] # Moving average for shut periods
    poma = opma / (opma + shma) # Moving average for Popen
    return opma, shma, poma

def filter_risetime(fc):
    return 0.3321 / fc

def amplitudes_openings_longer_Tr(rec, fc, n=2):
    all_resolved_ops = np.array(rec.rtint)[np.where( np.fabs(np.asarray(rec.rampl)) > 0.0)]
    all_resolved_opamp = np.array(rec.rampl)[np.where( np.fabs(np.asarray(rec.rampl)) > 0.0)]
    #long_ops = all_resolved_ops[np.where( all_resolved_ops > filter_risetime(fc))]
    long_opamp = all_resolved_opamp[np.where( all_resolved_ops > n * filter_risetime(fc))]
    return np.absolute(long_opamp)

##### fitting exponential pdf's ###############################################
def myexp(theta, X):
    tau, area = _theta_unsqueeze(theta)
    y = np.array([])
    for t in np.nditer(np.asarray(X)):
        y = np.append(y, np.sum((area / tau) * np.exp(-t / tau)))
    return y

def _theta_unsqueeze(theta):
    tau, area = np.split(np.asarray(theta), [int(math.ceil(len(theta) / 2))]) # pylint: disable=unbalanced-tuple-unpacking
    area = np.append(area, 1 - np.sum(area))
    return tau, area

def _log_likelihood_exponential_pdf(tau, area, X):
    d = np.sum( area * (np.exp(-min(X) / tau) - np.exp(-max(X)/ tau)))
    if d < 1.e-37:
        print (' ERROR in EXPLIK: d = ', d)
    s = 0.0
    for t in np.nditer(np.asarray(X)):
        s -= math.log(np.sum((area / tau) * np.exp(-t / tau)))
    return s + len(X) * math.log(d)

def LLexp(theta, X): # wrapper for log likelihood of exponential pdf
    tau, area = _theta_unsqueeze(theta)
    tau[tau < 1.0e-30] = 1e-8
    area[area > 1.0] = 0.99999
    area[area < 0.0] = 1e-6
    if np.sum(area[:-1]) >= 1: 
        area[:-1] = 0.99 * area[:-1] / np.sum(area[:-1])
    area[-1] = 1 - np.sum(area[:-1])    
    return _log_likelihood_exponential_pdf(tau, area, X)

def _predicted_number_per_component(theta, X): # predicted # of intervals per component
    tau, area = _theta_unsqueeze(theta)
    p1 = np.sum(area * np.exp(-min(X) / tau))  #Prob(obs>ylow)
    p2 = np.sum(area * np.exp(-max(X) / tau))  #Prob(obs>yhigh)
    antrue = len(X) / (p1 - p2)
    en = antrue * area
    enout = [antrue * (1. - p1), antrue * p2]
    return en, enout

def print_exps(theta, X):
    tau, area = _theta_unsqueeze(theta)
    numb, numout = _predicted_number_per_component(theta, X)
    for ta, ar, nu in zip(tau, area, numb):
        print('Tau = {0:.6f}; lambda (1/s)= {1:.6f}'.format(ta, 1.0 / ta))
        print('Area= {0:.6f}; predicted number of intervals = {1:.3f};'.format(ar, nu) + 
              'amplitude (1/s) = {0:.3f}'.format(ar / ta))
    print('\nOverall mean = {0:.6f}'.format(np.sum(area * tau)))
    print('Predicted true number of events = ', np.sum(numb))
    print('Number of fitted = ', len(X))
    print('Predicted number below Ylow = {0:.3f}; predicted number above Yhigh = {1:.3f}'.
          format(numout[0], numout[1]))

def fit_exponentials(tau, area, X):
    # TODO: check that area has one element less (len(tau)=len(area)+1)
    theta = tau + area
    print('Start LogLikelihood =', LLexp(theta, X))
    res = minimize(LLexp, theta, args=X, method='Nelder-Mead')
    print (res.message)
    print('Final LogLikelihood = {0:.6f}\n'.format(res.fun))
    return res
