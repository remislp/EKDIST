import math
import numpy as np
from numpy import linalg as nplin
from scipy.optimize import minimize, bisect

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

##### finding tcrit between exponentials ######################################

def expPDF_misclassified(tcrit, tau, area, comp):
    """ Calculate number and fraction of misclassified events after division into
    bursts by critical time, tcrit. """
    tfast, tslow = tau[:comp], tau[comp:]
    afast, aslow = area[:comp], area[comp:]
    # Number of misclassified.
    enf = np.sum(afast * np.exp(-tcrit / tfast))
    ens = np.sum(aslow * (1 - np.exp(-tcrit / tslow)))
    # Fraction misclassified.
    pf = enf / np.sum(afast)
    ps = ens / np.sum(aslow)
    return enf, ens, pf, ps

def expPDF_tcrit_DC(tcrit, tau, area, comp):
    """ """
    _, _, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
    return ps - pf

def expPDF_tcrit_CN(tcrit, tau, area, comp):
    """ """
    enf, ens, _, _ = expPDF_misclassified(tcrit, tau, area, comp)
    return ens - enf

def expPDF_tcrit_Jackson(tcrit, tau, area, comp):
    """ """
    tfast, tslow = tau[:comp], tau[comp:]
    afast, aslow = area[:comp], area[comp:]
    # Number of misclassified.
    enf = np.sum((afast / tfast) * np.exp(-tcrit / tfast))
    ens = np.sum((aslow / tslow) * np.exp(-tcrit / tslow))
    return enf - ens

def expPDF_misclassified_printout(tcrit, enf, ens, pf, ps):
    """ """
    return ('tcrit = {0:.5g} ms\n'.format(tcrit * 1000) +
        '% misclassified: short = {0:.5g};'.format(pf * 100) +
        ' long = {0:.5g}\n'.format(ps * 100) +
        '# misclassified (out of 100): short = {0:.5g};'.format(enf * 100) +
        ' long = {0:.5g}\n'.format(ens * 100) +
        'Total # misclassified (out of 100) = {0:.5g}\n'
        .format((enf + ens) * 100))

def get_tcrits(pars):
    tau, area = _theta_unsqueeze(pars)
    tcrits = np.empty((3, len(tau)-1))
    for i in range(len(tau)-1):
        print('\nCritical time between components {0:d} and {1:d}\n'.
                format(i+1, i+2) + '\nEqual % misclassified (DC criterion)')
        try:
            tcrit = bisect(expPDF_tcrit_DC, tau[i], tau[i+1], args=(tau, area, i+1))
            enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, i+1)
            print(expPDF_misclassified_printout(tcrit, enf, ens, pf, ps))
        except:
            print('Bisection with DC criterion failed.\n')
            tcrit = None
        tcrits[0, i] = tcrit
        
        print('Equal # misclassified (Clapham & Neher criterion)')
        try:
            tcrit = bisect(expPDF_tcrit_CN, tau[i], tau[i+1], args=(tau, area, i+1))
            enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, i+1)
            print(expPDF_misclassified_printout(tcrit, enf, ens, pf, ps))
        except:
            print('Bisection with Clapham & Neher criterion failed.\n')
            tcrit = None
        tcrits[1, i] = tcrit
            
        print('Minimum total # misclassified (Jackson et al criterion)')
        try:
            tcrit = bisect(expPDF_tcrit_Jackson, tau[i], tau[i+1], args=(tau, area, i+1))
            enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, i+1)
            print(expPDF_misclassified_printout(tcrit, enf, ens, pf, ps))
        except:
            print('\nBisection with Jackson et al criterion failed.')
            tcrit = None
        tcrits[2, i] = tcrit
        
    print('\n\nSUMMARY of tcrit values:\nComponents\t\tDC\t\tC&N\t\tJackson\n')
    for i in range(len(tau)-1):
        print('{0:d} to {1:d} '.format(i+1, i+2) +
                '\t\t\t{0:.5g}'.format(tcrits[0, i] * 1000) +
                '\t\t{0:.5g}'.format(tcrits[1, i] * 1000) +
                '\t\t{0:.5g}\n'.format(tcrits[2, i] * 1000))
    return tcrits

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
    return _log_likelihood_exponential_pdf(tau, area, np.asarray(X))

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

##### calculate approximate SD ###########################################

def hessian(theta, LLfunc, args, delta_step=0.0001):
    """ """
    hess = np.zeros((theta.size, theta.size))
    deltas = __optimal_deltas(theta, LLfunc, args, delta_step)
    # Diagonal elements of Hessian
    coe11 = np.array([theta.copy(), ] * theta.size) + np.diag(deltas)
    coe33 = np.array([theta.copy(), ] * theta.size) - np.diag(deltas)
    for i in range(theta.size):
        hess[i, i] = ((LLfunc(coe11[i], args) - 
            2.0 * LLfunc(theta, args) +
            LLfunc(coe33[i], args)) / (deltas[i]  ** 2))
    # Non diagonal elements of Hessian
    for i in range(theta.size):
        for j in range(theta.size):
            coe1, coe2, coe3, coe4 = theta.copy(), theta.copy(), theta.copy(), theta.copy()
            if i != j:                
                coe1[i] += deltas[i]
                coe1[j] += deltas[j]
                coe2[i] += deltas[i]
                coe2[j] -= deltas[j]
                coe3[i] -= deltas[i]
                coe3[j] += deltas[j]
                coe4[i] -= deltas[i]
                coe4[j] -= deltas[j]
                hess[i, j] = ((
                    LLfunc(coe1, args) -
                    LLfunc(coe2, args) -
                    LLfunc(coe3, args) +
                    LLfunc(coe4, args)) /
                    (4 * deltas[i] * deltas[j]))
    return hess

def __tune_deltas(theta, func, args, Lcrit, deltas, increase=True):
    factor = [1, 2] if increase else [-1, 0.5]
    count = 0
    while factor[0] * func(theta + deltas, args) < factor[0] * Lcrit and count < 100:
        deltas *= factor[1]
        count += 1
    return deltas

def __optimal_deltas(theta, LLfunc, args, step_factor=0.0001):
    Lcrit = LLfunc(theta, args) + math.fabs(LLfunc(theta, args) * 0.005)
    deltas = step_factor * theta
    L = LLfunc(theta + deltas, args)
    if L < Lcrit:
        deltas = __tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=True)
    elif L > Lcrit:
        deltas = __tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=False)
    return deltas

def covariance_matrix(theta, func, args, weightmode=1):
    """ """
    cov = nplin.inv(np.array(hessian(theta, func, args)))
#    if weightmode == 1:
#        errvar = SSD(theta, (func, args))[0] / (args[0].size - theta.size)
#    else:
#        errvar = 1.0
    return cov #* errvar

def correlation_matrix(covar):
    correl = np.zeros((len(covar),len(covar)))
    for i1 in range(len(covar)):
        for j1 in range(len(covar)):
            correl[i1,j1] = (covar[i1,j1] / 
                np.sqrt(np.multiply(covar[i1,i1],covar[j1,j1])))
    return correl

def approximate_SD(theta, func, arg, delta_step=0.0001):
    hess = hessian(theta, func, arg, delta_step)
    covariance = nplin.inv(hess)
    sd = np.sqrt(covariance.diagonal())
    __print_exps_with_errs(theta, sd)
    return sd

def __errs_unsqueeze(sd):
    sd = np.asarray(sd)
    tsd, asd = np.split(sd, [int(math.ceil(len(sd) / 2))]) # pylint: disable=unbalanced-tuple-unpacking
    asd = np.append(asd, asd[-1])
    return tsd, asd

def __print_exps_with_errs(theta, apprSD):
    tau, area = _theta_unsqueeze(theta)
    tsd, asd = __errs_unsqueeze(apprSD)
    for ta, ar, td, ad in zip(tau, area, tsd, asd):
        print('Tau = {0:.6f}; approximate SD = {1:.6f}'.format(ta, td))
        print('Area= {0:.6f}; approximate SD = {1:.6f}'.format(ar, ad))

##########

