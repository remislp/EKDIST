
import os, sys
import math
from scipy.optimize import minimize, bisect
import numpy as np
import numpy.testing as npt
from numpy import linalg as nplin

from ekdist import eklib

class TestErrorCalculation:
    def setUp(self):
        self.infile = "./tests/intervals.txt"
        self.intervals = np.loadtxt(self.infile)
        self.tau, self.area = [0.036, 1.1], [0.20]
        self.theta = self.tau + self.area
        self.epdf = eklib.ExponentialPDF(self.tau, self.area)
        self.res = minimize(self.epdf.LL, self.epdf.theta, 
                            args=self.intervals, 
                            method='Nelder-Mead')
        self.hess = eklib.hessian(self.res.x, self.epdf.LL, self.intervals)
        #self.cov = eklib.covariance_matrix(self.theta, eklib.LLexp, self.intervals)
        self.cov = nplin.inv(self.hess)
        self.appSD = np.sqrt(self.cov.diagonal())
        self.corr = eklib.correlation_matrix(self.cov)

    def test_infile_exists(self):
        assert os.path.isfile(self.infile)

    def test_interval_number(self):
        assert len(self.intervals)  == 125

    def test_expLogLik(self):
        npt.assert_almost_equal(self.epdf.LL(self.theta, self.intervals), 87.31806715582867)

    def test_estimates(self):
        npt.assert_almost_equal(self.res.x, np.array([0.03700718, 1.07302608, 0.19874548]))

    def test_hessian(self):
        npt.assert_almost_equal(self.hess, [[8475.49606292, -92.21425006, -626.06648917],
                                            [ -92.21425006,  71.5478832,   -36.17166531],
                                            [-626.06648917, -36.17166531,  501.13966047]])
    def test_covariance(self):
        npt.assert_almost_equal(self.cov, [[0.00013478, 0.00026864, 0.00018777],
                                           [0.00026864, 0.01504143, 0.00142128],
                                           [0.00018777, 0.00142128, 0.00233261]])

    def test_SD(self):
        npt.assert_almost_equal(self.appSD, [0.01160948, 0.1226435,  0.04829715])

    def test_correlations(self):
        npt.assert_almost_equal(self.corr, np.array([[1.        , 0.18867387, 0.33487997],
                                                     [0.18867387, 1.        , 0.23994593],
                                                     [0.33487997, 0.23994593, 1.        ]]))



    