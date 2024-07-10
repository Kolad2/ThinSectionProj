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


from .distrebutions import *
import SE



def FPM(x,a,scale):
    mA = 1/A(a)
    if x >= scale:
        return 1 - (mA/(a-1))*(x / scale)**(1 - a)
    else:
        return (mA/a)*np.exp(a)*(1 - np.exp(-a * (x / scale)))

def pareto(x, a, xmin):
    return ((a-1)/xmin)*((x/xmin)**(-a))


def get_ecdf(X, xmin=None):
    if xmin is not None:
        X = np.append(X, xmin)
    X, C = np.unique(X, return_counts=True)
    C[0] = 0
    F = np.cumsum(C)
    F = F / F[-1]
    return X, F


def lcdfgen(X, F, N):
    Fx = np.sort(np.random.rand(N))
    x = np.zeros(np.shape(Fx))
    k = 1
    for i in range(0, N):
        while Fx[i] > F[k]:
            k = k + 1
        x[i] = (X[k] - X[k - 1]) / (F[k] - F[k - 1]) * (Fx[i] - F[k - 1]) + X[k - 1]
    return x


def get_pv(ks ,v):
    X, F = get_ecdf(ks, xmin=None)
    xmin = X[0]
    xmax = X[-1]
    return np.interp(v, X, F)

class Targets:
    def __init__(self, X, xmin, xmax):
        self.SlnX = np.mean(np.log(X))
        self.SlnX2 = np.mean((np.log(X)) ** 2)
        self.SX = np.mean(X)
        self.X = X
        self.n = len(X)
        self.xmin = xmin
        self.xmax = xmax

    def lognorm(self, s, scale):
        dist = st.lognorm(s, 0, scale)
        Fmin = dist.cdf(self.xmin)
        Fmax = dist.cdf(self.xmax)
        mu = math.log(scale)
        part1 = -np.log(s) - self.SlnX2 / (2 * (s ** 2))
        part2 = (2 * mu * self.SlnX - mu ** 2) / (2 * (s ** 2))
        part3 = -np.log(Fmax - Fmin)
        return part1 + part2 + part3

    def expon(self, scale):
        Fmin = 1 - np.exp(self.xmin / scale)

        S = - self.SX / scale - np.log(scale) - xmin / scale
        return S

    def pareto(self, a, xmin):
        S = np.log(a-1) + (a-1)*np.log(xmin) - a*self.SlnX
        return S

    def weibull(self, alpha, scale):
        part1 = np.log(alpha/scale)
        part2 = (alpha - 1)*(self.SlnX - np.log(scale))
        part3 = -np.mean((self.X/scale)**alpha)
        part4 = -np.log(np.exp(-(self.xmin / scale)**alpha) - np.exp(-(self.xmax / scale)**alpha))
        return part1 + part2 + part3 + part4

    def paretomodif(self, a, xg):
        part1 = - np.log(xg) - np.log(A(a))
        part2 = -(a/self.n)*np.sum(self.X[self.X < xg]/xg - 1)
        part3 = -(a/self.n)*np.sum(np.log(self.X[self.X >= xg]/xg))
        Fmin = FPM(self.xmin, a, xg)
        Fmax = FPM(self.xmax, a, xg)
        part4 = -np.log(Fmax - Fmin)
        return part1 + part2 + part3 + part4



def GetF(S, xmin, xmax=10 ** 10):
    F_bins, F = np.unique(S, return_counts=True)
    F_bins = np.insert(F_bins,0,0)
    F_bins = np.append(F_bins, xmax)
    F = np.cumsum(F)
    F = np.insert(F, 0, 0)
    F = F / F[-1]
    return F_bins, F

def Getf(S, f_bins):
    f, _ = np.histogram(S, bins=f_bins, density=True)
    return f