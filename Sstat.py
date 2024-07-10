import os
import sys

import scipy.io
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
import math
from scipy.stats._continuous_distns import lognorm_gen
from scipy.ndimage import histogram
from scipy import stats as st
from scipy.special import factorial
from scipy.special import gamma
import scipy.stats as st
import scipy.optimize as opt
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from scipy.optimize import minimize, rosen, rosen_der

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
FileName = FileNames[10]

xmin = 20

matdict = scipy.io.loadmat("temp/StatisticRawData/" + FileName + "/" + FileName + "_1" + "_S.mat")
S = matdict['S'][0]
P = matdict['P'][0]
S = S[2:]
P = P[2:]
hS = np.empty(len(S), dtype=float)
for i in range(0,len(S)):
    hS[i] = S[i] + np.sum(st.uniform.rvs(size=P[i]))
hS = hS[hS > xmin]
u, c = np.unique(hS, return_counts=True)
bins = np.empty(len(u)+2, dtype=float)
counts = np.empty(len(c)+1, dtype=float)
bins[0] = 0; bins[1:-1] = u; bins[-1] = 10**10
c = np.cumsum(c)
counts[0] = 0; counts[1:] = c
F_0 = counts/counts[-1]

log_bins = xmin*np.logspace(0,10,160,base=2)
f_0,_ = np.histogram(hS, bins=log_bins, density=True)


def target_lognorm(s, scale, X, xmin):
    dist = st.lognorm(s, 0, scale)
    Fmin = dist.cdf(xmin)
    return -np.sum(np.log(dist.pdf(X)/(1 - Fmin)))


def target_expon(scale, X, xmin):
    Fmin = 1 - np.exp(xmin/scale)
    S = -(np.sum(-X/scale) - len(X)*np.log(scale) - len(X)*np.log(1 - Fmin))
    return S

def f_lognorm(x):
    return target_lognorm(x[0], x[1], hS, xmin)

def f_expon(x):
    return target_expon(x, hS, xmin)

def GetThetaLognorm():
    theta = st.lognorm.fit(hS, floc=0)
    res = minimize(f_lognorm, [theta[0], theta[2]], method='Nelder-Mead', tol=1e-6)
    return res.x[0], 0, res.x[1]

def GetThetaExpon():
    theta = st.expon.fit(hS, floc=0)
    res = minimize(f_expon, theta[1], method='Nelder-Mead', tol=1e-6)
    return 0, res.x[0]


theta = [[] for i in range(0,2)]
dist = [[] for i in range(0,2)]
F = [[] for i in range(0,2)]
f = [[] for i in range(0,2)]
theta[0] = GetThetaLognorm()
theta[1] = GetThetaExpon();print(theta[1])


dist[0] = st.lognorm(theta[0][0], 0, theta[0][2])
dist[1] = st.expon(0, theta[1][1])

def get_cdfcor(dist, X, xmin):
    return (dist.cdf(X) - dist.cdf(xmin)) / (1 - dist.cdf(xmin))

for i in range(0,2):
    F[i] = get_cdfcor(dist[i], log_bins, xmin)
    f[i] = (dist[i].pdf(log_bins))/(1 - dist[i].cdf(xmin))




D = np.max(np.abs(F_0[1:] - get_cdfcor(dist[0], u, xmin)))


pv = st.kstwo(len(hS), loc=0, scale=1).cdf(D)
print(D)
print(pv)

fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
ax[0].set_xscale('log')
ax[0].stairs(F_0, bins, fill=True)
ax[0].plot(log_bins, F[0],color='b')
ax[0].plot(log_bins, F[1],color='b')

ax[1].set_xscale('log')
#ax[1].set_yscale('log')
ax[1].stairs(f_0, log_bins, fill=True)
ax[1].plot(log_bins, f[0],color='b')
ax[1].plot(log_bins, f[1],color='r')
plt.show()



exit()
pl = 2.5
pl2 = (pl)**2

"""
log_bins = np.logspace(0,30,30,base=2)
f_0, bins = np.histogram(S,bins= log_bins, density=False)
dstr0 = st.rv_histogram((f_0, bins), density=False)
y = dstr0.cdf(log_bins)
print(y)
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
#ax[0].plot(log_bins,y)
ax[0].stairs(f_0, log_bins, fill=True)
plt.show()
exit()
"""



S = S + np.sqrt(S)
mask = S > 10
S = S[mask]

log_bins = np.logspace(1,20,20,base=2)
counts, bins = np.histogram(S,bins=log_bins)
Dl = bins[1:] - bins[0:-1]
f_0 = counts/(Dl*sum(counts))
N = len(S)

theta_1 = st.lognorm.fit(S,floc=0); print(len(theta_1))
theta_2 = st.expon.fit(S,floc=0); print(len(theta_2))
theta_3 = st.powerlaw.fit(S,floc=0); print(len(theta_3))
theta_4 = st.pareto.fit(S,floc=0); print(len(theta_4))
theta_5 = st.powerlognorm.fit(S,floc=0); print(len(theta_5))

dstr = [[] for i in range(0,5)]
dstr_0 = st.rv_histogram((counts, bins), density=False)
dstr[0] = st.lognorm(theta_1[0], theta_1[1], theta_1[2])
dstr[1] = st.expon(theta_2[0], theta_2[1])
dstr[2] = st.powerlaw(theta_3[0], theta_3[1], theta_3[2])
dstr[3] = st.pareto(theta_4[0], theta_4[1], theta_4[2])
dstr[4] = st.powerlognorm(theta_5[0],theta_5[1],theta_5[2],theta_5[3])
rng = np.random.default_rng()


K = [[] for i in range(0,5)]
S_G = [[] for i in range(0,5)]
f_G = [[] for i in range(0,5)]
for i in range(0,10):
    print(i)
    S_G0 = dstr_0.rvs(size=N, random_state=rng)
    for i in range(0, 5):
        S_G[i] = dstr[i].rvs(size=N, random_state=rng)
    S_GU = st.uniform.rvs(size=N) * 2 - 1
    S_G0 = S_G0 + 0.1*S_G0*S_GU
    S_G0 = S_G0[(S_G0 > min(S)) & (S_G0 < max(S))]
    f_G0, bins = np.histogram(S_G0, bins=bins, density=True)
    for i in range(0,5):
        S_G[i] = S_G[i][(S_G[i] > min(S)) & (S_G[i] < max(S))]
        f_G[i], bins = np.histogram(S_G[i], bins=bins, density=True)
        K[i].append(max(abs(np.cumsum(f_G0 * Dl) - np.cumsum(f_G[i] * Dl))))

linbins = [[] for i in range(0,5)]
f_K = [[] for i in range(0,5)]
for i in range(0,5):
    K[i] = np.array(K[i])
    linbins[i] = np.linspace(K[i].mean()-3*K[i].std(),K[i].mean()+3*K[i].std(),50)
    f_K[i], _ = np.histogram(K[i],bins=linbins[i],density=True)


KS1 = st.ks_1samp(S, dstr[0].cdf)
KS2 = st.ks_1samp(S, dstr[1].cdf)
KS3 = st.ks_1samp(S, dstr[2].cdf)
KS4 = st.ks_1samp(S, dstr[3].cdf)
KS5 = st.ks_1samp(S, dstr[4].cdf)
print(KS1)
print(KS2)
print(KS3)
print(KS4)
print(KS5)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
#ax[0].set_xscale('log')
#ax[0].set_yscale('log')
ax[0].stairs(np.cumsum(f_K[0]*6*K[0].std()/50), linbins[0], fill=False, label='lognorm',linewidth=2.0,color='red')
ax[0].stairs(np.cumsum(f_K[1]*6*K[1].std()/50), linbins[1], fill=False, label='expon',linewidth=2.0,color='blue')
ax[0].stairs(np.cumsum(f_K[2]*6*K[2].std()/50), linbins[2], fill=False, label='power',linewidth=2.0,color='y')
ax[0].stairs(np.cumsum(f_K[3]*6*K[3].std()/50), linbins[3], fill=False, label='pareto',linewidth=2.0,color='g')
ax[0].stairs(np.cumsum(f_K[4]*6*K[4].std()/50), linbins[4], fill=False, label='powerlognorm',linewidth=2.0,color='violet')
#ax[0].stairs(f_G1, log_bins, fill=False)
ax[0].set_title(FileName)
ax[0].legend()
#ax[0].set_xlim((10**2, 10**2))
#plt.show()
fig.savefig("temp/" + FileName + "_pf_S.png")


f_1 = dstr[0].pdf(log_bins)
f_2 = dstr[1].pdf(log_bins)
f_3 = dstr[2].pdf(log_bins)
f_4 = dstr[3].pdf(log_bins)
f_5 = dstr[4].pdf(log_bins)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].stairs(f_0, log_bins, fill=True)
ax[0].plot(log_bins, f_1, label='lognorm',linewidth=2.0,color='red')
ax[0].plot(log_bins, f_2, label='expon',linewidth=2.0,color='blue')
ax[0].plot(log_bins, f_3, label='power',linewidth=2.0,color='y')
ax[0].plot(log_bins, f_4, label='pareto',linewidth=2.0,color='g')
ax[0].plot(log_bins, f_5, label='powerlognorm',linewidth=2.0,color='brown')
ax[0].set_title(FileName)
ax[0].legend()
#ax[0].set_xlim((10**2, 10**2))
ax[0].set_ylim((10**(-10)), 10**(-2))
fig.savefig("temp/" + FileName + "_log_pdf_S.png")
plt.show()