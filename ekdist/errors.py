import math
import numpy as np
from numpy import linalg as nplin

class ApproximateSD:
    def __init__(self, theta, func, arg, delta_step=0.0001):
        self.hessian = self.hessian_matrix(theta, func, arg, delta_step)
        self.covariance = nplin.inv(self.hessian)
        self.sd = np.sqrt(self.covariance.diagonal())
        self.correlations = self.correlation_matrix(self.covariance)

    def hessian_matrix(self, theta, LLfunc, args, delta_step=0.0001):
        """ """
        hess = np.zeros((theta.size, theta.size))
        deltas = self.__optimal_deltas(theta, LLfunc, args, delta_step)
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

    def __tune_deltas(self, theta, func, args, Lcrit, deltas, increase=True):
        factor = [1, 2] if increase else [-1, 0.5]
        count = 0
        while factor[0] * func(theta + deltas, args) < factor[0] * Lcrit and count < 100:
            deltas *= factor[1]
            count += 1
        return deltas

    def __optimal_deltas(self, theta, LLfunc, args, step_factor=0.0001):
        Lcrit = LLfunc(theta, args) + math.fabs(LLfunc(theta, args) * 0.005)
        deltas = step_factor * theta
        L = LLfunc(theta + deltas, args)
        if L < Lcrit:
            deltas = self.__tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=True)
        elif L > Lcrit:
            deltas = self.__tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=False)
        return deltas

    def correlation_matrix(self, covar):
        correl = np.zeros((len(covar),len(covar)))
        for i1 in range(len(covar)):
            for j1 in range(len(covar)):
                correl[i1,j1] = (covar[i1,j1] / 
                    np.sqrt(np.multiply(covar[i1,i1],covar[j1,j1])))
        return correl

