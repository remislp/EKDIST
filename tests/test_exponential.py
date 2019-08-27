import os, sys
import math
from scipy.optimize import minimize, bisect
import numpy as np
import numpy.testing as npt
from numpy import linalg as nplin

from ekdist import exponentials

def test_area_none():
    tau_in = [0.1, 1.0]
    epdf = exponentials.ExponentialPDF(tau=tau_in)
    area = np.ones(2) / 2
    print(epdf.area)
    npt.assert_almost_equal(epdf.area, area)
    pars = np.concatenate((np.asarray(tau_in), area))
    print(epdf.pars)
    npt.assert_almost_equal(epdf.pars, pars)
    print(epdf.fixed)
    fixed = [False, False, False, True]
    assert fixed == epdf.fixed
    print(epdf.theta)
    theta = pars[: -1]
    npt.assert_almost_equal(epdf.theta, theta)

def test_insert_theta():
    tau_in = [0.1, 1.0]
    epdf = exponentials.ExponentialPDF(tau=tau_in)
    theta_in = [0.2, 2., 0.3]
    epdf.theta = theta_in
    print(epdf.theta)
    print(epdf.pars)
    pars_in = [0.2, 2., 0.3, 0.7]
    npt.assert_almost_equal(epdf.pars, pars_in)

def test_fixed_pars():
    tau_in = [0.1, 1.0]
    epdf = exponentials.ExponentialPDF(tau=tau_in)
    epdf.fixed[1] = True
    fixed = [False, True, False, True]
    assert fixed == epdf.fixed
    theta_in = [0.2, 0.3]
    epdf.theta = theta_in
    print(epdf.theta)
    print(epdf.pars)
    pars_in = [0.2, 1., 0.3, 0.7]
    npt.assert_almost_equal(epdf.pars, pars_in)
    npt.assert_almost_equal(epdf.theta, theta_in)
