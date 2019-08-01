import math
import numpy as np
from numpy import linalg as nplin
from scipy.optimize import minimize, bisect

class ExponentialPDF(object):
    def __init__(self, tau, area):
        """        """
        self.eqname = 'exponential pdf'
        self.tau = np.asarray(tau)
        if len(tau) == len(area):
            self.area = np.asarray(area)
        else:
            self.area = np.append(np.asarray(area), 1 - np.sum(area))
        self.pars = np.concatenate((self.tau, self.area))
        self.ncomp = len(tau)
        #self.guess = None
        self._theta = None
        self.fixed = [False, False] * self.ncomp
        self.fixed[-1] = True
        #self.names = ['tau', 'area']
        
    def exp(self, theta, X):
        tau, area = self.__theta_unsqueeze(theta)
        X = np.asarray(X)
        y = np.array([])
        for t in np.nditer(X):
            y = np.append(y, np.sum((area / tau) * np.exp(-t / tau)))
        return y

    def LL(self, theta, X):
        tau, area = self.__theta_unsqueeze(theta)
        tau[tau < 1.0e-30] = 1e-8
        area[area > 1.0] = 0.99999
        area[area < 0.0] = 1e-6
        if np.sum(area[:-1]) >= 1: 
            area[:-1] = 0.99 * area[:-1] / np.sum(area[:-1])
        area[-1] = 1 - np.sum(area[:-1])    
        return self.__log_likelihood_exponential_pdf(tau, area, np.asarray(X))

    def __log_likelihood_exponential_pdf(self, tau, area, X):
        d = np.sum( area * (np.exp(-min(X) / tau) - np.exp(-max(X)/ tau)))
        if d < 1.e-37:
            print (' ERROR in EXPLIK: d = ', d)
        s = 0.0
        for t in np.nditer(np.asarray(X)):
            s -= math.log(np.sum((area / tau) * np.exp(-t / tau)))
        return s + len(X) * math.log(d)

    def _set_theta(self, theta):
        if self.pars is None:
            self.pars = np.zeros(len(theta) + 1)
        for each in np.nonzero(self.fixed)[0]:   
            theta = np.insert(theta, each, self.pars[each])
        self.pars = theta
    def _get_theta(self):
        theta = self.pars[np.nonzero(np.invert(self.fixed))[0]]
        if isinstance(theta, float):
            theta = np.array([theta])
        return theta
    theta = property(_get_theta, _set_theta)

    def __theta_unsqueeze(self, theta):
        tau, area = np.split(np.asarray(theta), [int(math.ceil(len(theta) / 2))]) # pylint: disable=unbalanced-tuple-unpacking
        area = np.append(area, 1 - np.sum(area))
        return tau, area

    def __predicted_number_per_component(self, X): # predicted # of intervals per component
        p1 = np.sum(self.area * np.exp(-min(X) / self.tau))  #Prob(obs>ylow)
        p2 = np.sum(self.area * np.exp(-max(X) / self.tau))  #Prob(obs>yhigh)
        antrue = len(X) / (p1 - p2)
        en = antrue * self.area
        enout = [antrue * (1. - p1), antrue * p2]
        return en, enout

    def __print_exps(self, X):
        numb, numout = self.__predicted_number_per_component(X)
        for ta, ar, nu in zip(self.tau, self.area, numb):
            print('Tau = {0:.6f}; lambda (1/s)= {1:.6f}'.format(ta, 1.0 / ta))
            print('Area= {0:.6f}; predicted number of intervals = {1:.3f};'.format(ar, nu) + 
                'amplitude (1/s) = {0:.3f}'.format(ar / ta))
        print('\nOverall mean = {0:.6f}'.format(np.sum(self.area * self.tau)))
        print('Predicted true number of events = ', np.sum(numb))
        print('Number of fitted = ', len(X))
        print('Predicted number below Ylow = {0:.3f}; predicted number above Yhigh = {1:.3f}'.
            format(numout[0], numout[1]))

    def fit(self, X):
        print('Start LogLikelihood =', self.LL(self.theta, X))
        res = minimize(self.LL, self.theta, args=X, method='Nelder-Mead')
        print (res.message)
        print('Final LogLikelihood = {0:.6f}\n'.format(res.fun))
        self.tau, self.area = self.__theta_unsqueeze(res.x)
        self._set_theta(res.x)
        self.__print_exps(X)


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

def __theta_unsqueeze(theta):
    tau, area = np.split(np.asarray(theta), [int(math.ceil(len(theta) / 2))]) # pylint: disable=unbalanced-tuple-unpacking
    area = np.append(area, 1 - np.sum(area))
    return tau, area

def get_tcrits(pars):
    tau, area = __theta_unsqueeze(pars)
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


