import os
import sys
import scipy.io
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import math
import scipy as sp
from scipy.ndimage import histogram
from scipy import stats as st
from scipy.special import factorial
from scipy.special import gamma
import scipy.optimize as opt
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('error')

from .distrebutions import *
from .StatisticEstimation import *

class MLE:
    def __init__(self, X, xmin, xmax):
        self.TG = Targets(X, xmin, xmax)
        self.X = X
        self.xmin = xmin
        self.xmax = xmax

    def Lognorm(self):
        theta = st.lognorm.fit(self.X, floc=0)
        res = minimize(lambda x: -self.TG.lognorm(x[0], x[1]),
                       [theta[0], theta[2]],
                       bounds=((1e-3, None), (1e-3, self.xmax)),
                       method='Nelder-Mead', tol=1e-3)
        return res.x[0], 0, res.x[1]

    def Weibull(self):
        theta = st.weibull_min.fit(self.X, floc=0)
        res = minimize(lambda x: -self.TG.weibull(x[0], x[1]),
                       [theta[0], theta[2]], bounds=((1e-3, None), (1e-3, self.xmax)),
                       method='Nelder-Mead', tol=1e-3)
        return res.x[0], 0, res.x[1]

    def ParetoMod(self):
        a = 1 + 1 / (np.mean(np.log(self.X)) - np.log(self.xmin))
        res = minimize(lambda x: -self.TG.paretomod(x[0], x[1]),
                       [a, self.xmin], bounds=((1 + 1e-3, None), (self.xmin, self.xmax)),
                       method='Nelder-Mead', tol=1e-3)
        return res.x[0], 0, res.x[1]


def GetThetaExpon(X, xmin, xmax):
    theta = st.expon.fit(X, floc=0)
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.expon(x),
                   theta[1],
                   method='Nelder-Mead', tol=1e-3)
    return 0, res.x[0]

def GetThetaPareto(X, xmin, xmax):
    a = 1 + 1 / (np.mean(np.log(X)) - np.log(xmin))
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.pareto(x[0], x[1]),
                   [2, xmin/2], bounds=((1+1e-3, None), (0, xmin)),
                   method='Nelder-Mead', tol=1e-3)
    return res.x[0], 0, res.x[1]