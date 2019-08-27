import math
import numpy as np
from numpy import linalg as nplin
from scipy.optimize import minimize, bisect

class ExponentialPDF(object):
    def __init__(self, tau, area=None):
        """        """
        self.eqname = 'exponential pdf'
        self.tau = np.asarray(tau)
        self.ncomps = len(self.tau)
        self.area = area
        if area is None:
            self.area = np.ones(self.ncomps) / self.ncomps
        elif len(self.tau) == 1:
            self.area = np.array([1])
        elif len(tau) == (len(area) + 1):
             self.area = np.append(np.asarray(area), 1 - np.sum(area))
        elif len(tau) == len(area):
            self.area = np.asarray(area)
            self.area[-1] = 1. - np.sum(area)
        else:
            self.area = np.ones(self.ncomps) / self.ncomps

        self.pars = np.concatenate((self.tau, self.area))
        self.fixed = [False, False] * self.ncomps
        self.fixed[-1] = True        
        self._theta = self._get_theta()
        
        self.names = ['tau'] * self.ncomps + ['area'] * self.ncomps
        self.tcrits = np.empty((3, self.ncomps - 1))
        
    def exp(self, theta, X):
        self._set_theta(theta)
        #tau, area = self.__theta_unsqueeze(theta)
        X = np.asarray(X)
        y = np.array([])
        for t in np.nditer(X):
            y = np.append(y, np.sum((self.area / self.tau) * np.exp(-t / self.tau)))
        return y

    def LL(self, theta, X):
        self._set_theta(theta)
        #tau, area = self.__theta_unsqueeze(theta)
        #tau[tau < 1.0e-30] = 1e-8
        #area[area > 1.0] = 0.99999
        #area[area < 0.0] = 1e-6
        #if np.sum(area[:-1]) >= 1: 
        #    area[:-1] = 0.99 * area[:-1] / np.sum(area[:-1])
        #area[-1] = 1 - np.sum(area[:-1])    
        return self.__log_likelihood_exponential_pdf(self.tau, self.area, np.asarray(X))

    def __log_likelihood_exponential_pdf(self, tau, area, X):
        d = np.sum( area * (np.exp(-min(X) / tau) - np.exp(-max(X)/ tau)))
        if d < 1.e-37:
            print (' ERROR in EXPLIK: d = ', d)
        s = 0.0
        for t in np.nditer(np.asarray(X)):
            s -= math.log(np.sum((area / tau) * np.exp(-t / tau)))
        return s + len(X) * math.log(d)

    def _set_theta(self, theta):
        #print("set: theta=", theta)
        #print("set: pars=", self.pars)
        #print("set: nonzeros= ", np.nonzero(self.fixed))
        for each in np.nonzero(self.fixed)[0]:   
        #    print("set: each=", each)
            theta = np.insert(theta, each, self.pars[each])
        #    print("set: thetaeach=", theta)
        self.tau, self.area = np.split(np.asarray(theta), 2) # pylint: disable=unbalanced-tuple-unpacking
        self.area[-1] = 1. - np.sum(self.area[: -1])
        self.pars = np.concatenate((self.tau, self.area))
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

    def get_tcrits(self, verbose=True):
        tc = Tcrit(self.tau, self.area, verbose)
        self.tcrits = tc.tcrits


class Tcrit(object):
    def __init__(self, tau, area, verbose=False):
        """ Find tcrit between exponentials """
        self.tau = tau
        self.area = area
        self.tcrits = {}
        self.verbose = verbose
        self.__get_tcrits()
        if self.verbose: self.__print_summary()

    def __get_tcrits(self):

        if self.verbose: print('\nEqual % misclassified (DC criterion)')
        self.tcrits['DC'] = self.__calculate_tcrits(self.__tcrit_DC)

        if self.verbose: print('\nEqual # misclassified (Clapham & Neher criterion)')
        self.tcrits['C&N'] = self.__calculate_tcrits(self.__tcrit_CN)

        if self.verbose: print('\nMinimum total # misclassified (Jackson et al criterion)')
        self.tcrits['Jackson'] = self.__calculate_tcrits(self.__tcrit_Jackson)
        
    def __calculate_tcrits(self, func):
        tcrits = []
        for i in range(len(self.tau)-1):
            if self.verbose:
                print('Critical time between components {0:d} and {1:d}'.format(i+1, i+2))
            try:
                tcrit = bisect(func, self.tau[i], self.tau[i+1], args=(self.tau, self.area, i+1))
                enf, ens, pf, ps = self.__misclassified(tcrit, i+1)
                if self.verbose: self.__print_misclassified(tcrit, enf, ens, pf, ps)
            except:
                if self.verbose: print('Bisection failed.')
                tcrit = None
            tcrits.append(tcrit)
        return tcrits

    def __misclassified(self, tcrit, comp):
        """ Calculate number and fraction of misclassified events after division into
        bursts by critical time, tcrit. """
        tfast, tslow = self.tau[:comp], self.tau[comp:]
        afast, aslow = self.area[:comp], self.area[comp:]
        # Number of misclassified.
        enf = np.sum(afast * np.exp(-tcrit / tfast))
        ens = np.sum(aslow * (1 - np.exp(-tcrit / tslow)))
        # Fraction misclassified.
        pf = enf / np.sum(afast)
        ps = ens / np.sum(aslow)
        return enf, ens, pf, ps

    def __tcrit_DC(self, tcrit, tau, area, comp):
        _, _, pf, ps = self.__misclassified(tcrit, comp)
        return ps - pf

    def __tcrit_CN(self, tcrit, tau, area, comp):
        enf, ens, _, _ = self.__misclassified(tcrit, comp)
        return ens - enf

    def __tcrit_Jackson(self, tcrit, tau, area, comp):
        tfast, tslow = tau[:comp], tau[comp:]
        afast, aslow = area[:comp], area[comp:]
        # Number of misclassified.
        enf = np.sum((afast / tfast) * np.exp(-tcrit / tfast))
        ens = np.sum((aslow / tslow) * np.exp(-tcrit / tslow))
        return enf - ens

    def __print_misclassified(self, tcrit, enf, ens, pf, ps):
        print ('tcrit = {0:.5g} ms\n'.format(tcrit * 1000) +
            '% misclassified: short = {0:.5g};'.format(pf * 100) +
            ' long = {0:.5g}\n'.format(ps * 100) +
            '# misclassified (out of 100): short = {0:.5g};'.format(enf * 100) +
            ' long = {0:.5g}\n'.format(ens * 100) +
            'Total # misclassified (out of 100) = {0:.5g}'
            .format((enf + ens) * 100))

    def __print_summary(self):
        print ('\nSUMMARY of tcrit values (in ms):\nComponents\t\tDC\t\tC&N\t\tJackson')
        for i in range(len(self.tau)-1):
            print('{0:d} to {1:d} '.format(i+1, i+2) +
                    '\t\t\t{0:.5g}'.format(self.tcrits['DC'][i] * 1000) +
                    '\t\t{0:.5g}'.format(self.tcrits['C&N'][i] * 1000) +
                    '\t\t{0:.5g}'.format(self.tcrits['Jackson'][i] * 1000))

