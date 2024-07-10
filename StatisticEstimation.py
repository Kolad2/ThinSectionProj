import os
import sys
import scipy.io
import PathCreator
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
import csv
import cv2
import pandas as pd

def A(a):
    return ((a - 1) * np.exp(a) + 1) / (a * (a - 1))

def paretomodif(X,a,xg):
    C = 1/(xg*A(a))
    Y = X.copy()
    mask1 = X >= xg
    mask2 = X < xg
    Y[mask1] = (X[mask1]/xg)**(-a)
    Y[mask2] = (np.exp((-a)*(X[mask2]/xg - 1)))
    return C*Y

def Fparetomodif(X,a,xg):
    mA = 1/A(a)
    X = np.array(X)
    Y = np.empty(X.shape)
    mask1 = X >= xg
    mask2 = X < xg
    Y[mask1] = 1 - (mA/(a-1))*(X[mask1]/xg)**(1-a)
    Y[mask2] = mA/a*np.exp(a)*(1 - np.exp(-a*(X[mask2]/xg)))
    return Y

def FPM(x,a,xg):
    mA = 1/A(a)
    if x >= xg:
        return 1 - (mA/(a-1))*(x/xg)**(1-a)
    else:
        return (mA/a)*np.exp(a)*(1 - np.exp(-a*(x/xg)))

def pareto(x, a, xmin):
    return ((a-1)/xmin)*((x/xmin)**(-a))

def weibull(x, alpha, scale):
    return (alpha/scale)*(x/scale)**(alpha-1)*np.exp(-(x/scale)**alpha)

def Fweibull(x, alpha, scale):
    return 1 - np.exp(-(x/scale)**alpha)

class Targets:
    def lognorm(self, s, scale):
        dist = st.lognorm(s, 0, scale)
        Fmin = dist.cdf(self.xmin)
        mu = math.log(scale)
        part1 = -np.log(s) - self.SlnX2 / (2 * (s ** 2))
        part2 = (2 * mu * self.SlnX - mu ** 2) / (2 * (s ** 2))
        part3 = -np.log(1 - Fmin)
        return part1 + part2 + part3

    def expon(self, scale):
        Fmin = 1 - np.exp(xmin / scale)
        S = - self.SX / scale - np.log(scale) - xmin / scale
        return S

    def pareto(self, a, xmin):
        print(a)
        S = np.log(a-1) + (a-1)*np.log(xmin) - a*self.SlnX
        return S

    def weibull(self, alpha, scale):
        part1 = np.log(alpha/scale)
        part2 = (alpha - 1)*(self.SlnX - np.log(scale))
        part3 = -np.mean((self.X/scale)**alpha)
        part4 = (self.xmin / scale)**alpha
        return part1 + part2 + part3 + part4

    def paretomodif(self, a, xg):
        part1 = - np.log(xg) - np.log(A(a))
        part2 = -(a/self.n)*np.sum(self.X[self.X < xg]/xg - 1)
        part3 = -(a/self.n)*np.sum(np.log(self.X[self.X >= xg]/xg))
        part4 = -np.log(1 - FPM(self.xmin, a, xg))
        return part1 + part2 + part3 + part4

    def __init__(self, X, xmin, xmax):
        self.SlnX = np.mean(np.log(X))
        self.SlnX2 = np.mean((np.log(X)) ** 2)
        self.SX = np.mean(X)
        self.X = X
        self.n = len(X)
        self.xmin = xmin
        self.xmax = xmax


def GetThetaLognorm(X, xmin, xmax):
    theta = st.lognorm.fit(X, floc=0)
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.lognorm(x[0], x[1]),
                   [theta[0], theta[2]],
                   bounds=((0, None), (0, None)),
                   method='Nelder-Mead', tol=1e-3)
    return res.x[0], 0, res.x[1]

def GetThetaExpon(X, xmin, xmax):
    theta = st.expon.fit(X, floc=0)
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.expon(x),
                   theta[1],
                   method='Nelder-Mead', tol=1e-3)
    return 0, res.x[0]

def GetThetaWeibull(X, xmin, xmax):
    theta = st.weibull_min.fit(X, floc=0)
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.weibull(x[0], x[1]),
                   [theta[0], theta[2]], bounds=((1e-3, None), (1e-3, None)),
                   method='Nelder-Mead', tol=1e-3)
    print(res.x)
    return res.x[0], 0, res.x[1]

def GetThetaPareto(X, xmin, xmax):
    a = 1 + 1 / (np.mean(np.log(X)) - np.log(xmin))
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.pareto(x[0], x[1]),
                   [2, xmin/2], bounds=((1+1e-3, None), (0, xmin)),
                   method='Nelder-Mead', tol=1e-3)
    return res.x[0], 0, res.x[1]

def GetThetaParetoModif(X, xmin, xmax):
    a = 1 + 1 / (np.mean(np.log(X)) - np.log(xmin))
    Tg = Targets(X, xmin, xmax)
    res = minimize(lambda x: -Tg.paretomodif(x[0], x[1]),
                   [2, xmin], bounds=((1+1e-3, None), (xmin, 100*xmin)),
                   method='Nelder-Mead', tol=1e-3)
    return res.x[0], 0, res.x[1]

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