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

    def pareto(self, a, xmin):
        S = np.log(a-1) + (a-1)*np.log(xmin) - a*self.SlnX
        return S

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

FileNames = ["B21-234a",    #0
             "B21-215b",    #1
             "B21-215a",    #2
             "B21-213b",    #3
             "B21-213a",    #4
             "B21-189b",    #5
             "B21-188b_2",  #6
             "B21-151b",    #7
             "B21-107b",    #8
             "64-3",        #9
             "15-2",        #10
             "15-1",        #11
             "B21-166b",    #12
             "B21-122a",    #13
             "B21-120a",    #14
             "B21-51a",     #15
             "B21-200b",    #16
             "B21-192b",    #17
             "19-5b"]       #18
FileNames = [FileNames[18]]

for FileName in FileNames:
    xmin = 20
    xmax = 10*10

    matdict = scipy.io.loadmat("temp/StatisticSintData/" + FileName + "/" + FileName + "_S.mat", squeeze_me=True)

    hS = matdict['S']
    P = matdict['P']
    F_0 = []
    f_0 = []
    F_0_bins = []
    f_bins = xmin*np.logspace(0,10,60,base=2)
    f_0_m = np.empty(len(f_bins))
    f_0_sgm = np.empty(len(f_bins))
    numinter = 10
    for j in range(0, numinter):
        hS[j] = hS[j][hS[j] > xmin]
        f_0.append(Getf(hS[j], f_bins))
        F_bins, F = GetF(hS[j], xmin, np.max(hS[j]))
        F = np.interp(f_bins, F_bins[0:-1], F)
        F_0.append(F)


    f_0_med = np.median(f_0, axis=0)
    f_0_m = np.mean(f_0, axis=0)
    f_0_sgm = np.sqrt(np.mean((f_0-f_0_m)**2, axis=0))
    f_0_low = np.quantile(f_0, 0.01, axis=0)
    f_0_height = np.quantile(f_0, 0.99, axis=0)
    F_0_max = np.quantile(F_0, 0.99, axis=0)
    F_0_min = np.quantile(F_0,0.01, axis=0)
    F_0_med = np.median(F_0, axis=0)

    theta = [[] for i in range(0,4)]
    dist = [[] for i in range(0,4)]
    F = [[] for i in range(0,4)]
    f = [[] for i in range(0,4)]

    full_hS = np.concatenate(hS)
    theta[0] = GetThetaLognorm(full_hS, xmin, max(hS[0]))
    theta[1] = GetThetaExpon(full_hS, xmin, max(hS[0]))
    theta[2] = GetThetaParetoModif(hS[0], xmin, max(hS[0]))
    print(theta[0])
    print(theta[1])
    print(theta[2])

    dist[0] = st.lognorm(theta[0][0], 0, theta[0][2])
    dist[1] = st.expon(0, theta[1][1])




    for i in range(0,2):
        F[i] = (dist[i].cdf(f_bins) - dist[i].cdf(xmin))/(1 - dist[i].cdf(xmin))
        f[i] = dist[i].pdf(f_bins)/(1 - dist[i].cdf(xmin))
    F[2] = (Fparetomodif(f_bins, theta[2][0], theta[2][2]) - Fparetomodif(xmin, theta[2][0], theta[2][2]))/(1 - FPM(xmin, theta[2][0], theta[2][2]))
    f[2] = paretomodif(f_bins, theta[2][0], theta[2][2])/(1 - FPM(xmin, theta[2][0], theta[2][2]))
    xgp = theta[2][2]

    print(sum(f_0_height*(f_bins[1:] - f_bins[0:-1])))
    print(sum(f_0_low*(f_bins[1:] - f_bins[0:-1])))
    print(sp.integrate.trapezoid(f[0], f_bins))
    print(sp.integrate.trapezoid(f[1], f_bins))
    print(sp.integrate.trapezoid(f[2], f_bins))

    fig = plt.figure(figsize=(20, 10))
    ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
    ax[0].set_xscale('log')
    ax[0].fill_between(f_bins, F_0_min, F_0_max, alpha=.2, linewidth=0, color='red')
    ax[0].plot(f_bins, F[0], color='black')
    ax[0].plot(f_bins, F[1], color='black', linestyle='dashdot')
    ax[0].plot(f_bins, F[2], color='black', linestyle='dashed')
    ax[0].set_xlim((f_bins[0], f_bins[-1]))
    ax[0].set_ylim((-0.01, 1.01))
    # %%
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].fill_between(f_bins[0:-1], f_0_low, np.insert(f_0_height[0:-1], 0, f_0_height[0]), alpha=0.8, linewidth=0, color='red')
    ax[1].plot(f_bins, f[0],
               color='black',label="lognormal")
    ax[1].plot(f_bins, f[1],
           color='black',linestyle='dashdot',label="expon")
    ax[1].plot(f_bins[f_bins > xgp], f[2][f_bins > xgp],
           color='black',linestyle='dashed',label="pareto " + str(int(xgp)))
    ax[1].set_ylim((f_0_low[-1], f_0_height[0]))
    ax[1].set_xlim((f_bins[0], f_bins[-2]))
    ax[1].legend()
    fig.suptitle(FileName, fontsize=16)
    fig.savefig("temp/" + FileName + "_pf_S.png")
    #plt.show()


exit()