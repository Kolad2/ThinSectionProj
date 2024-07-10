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
import SE.StatisticEstimation as SE

import SE

StatisticPath = "/media/kolad/HardDisk/StatisticData/"
Path0 = "/media/kolad/HardDisk/ThinSection"

FileNames = os.listdir(StatisticPath + "StatisticCorData")


def GetData(FileName):
    matdict = sp.io.loadmat(StatisticPath + "StatisticCorData/" + FileName + "/" + FileName + "_S.mat", squeeze_me=True)
    hS = matdict['S']
    P = matdict['P']
    return hS, P

with open(StatisticPath + "StatisticResult.csv", 'r', encoding='UTF8', newline='') as f:
    rows_stats = list(csv.reader(f, delimiter=',', quotechar='|'))
    tb_stat = pd.DataFrame(rows_stats[1:], columns=rows_stats[0])
    del rows_stats


Theta = {
    "Name": [],
    "lognorm": {"boolean": [], "sigma": [], "lambda": []},
    "weibull": {"boolean": [], "alpha": [], "lambda": []},
    "paretomod": {"boolean": [], "alpha": [], "lambda": []}}

for i in range(0, len(tb_stat["Номер Образца"])):
    h = tb_stat["Номер Образца"].values[i]
    print(h)


xmin = 20
numinter = 500

for i0, FileName in enumerate(FileNames):
    print(FileName,"(", i0+1 , "/", len(FileNames), ")")

    # if len(tb_stat[tb_stat["Номер Образца"] == FileName]) > 0:
    #     print("Статистика образца найдена, пропуск")
    #     continue

    hS, P = GetData(FileName)

    Path_dir = Path0 + "/" + FileName + "/"
    F_0 = []
    f_0 = []
    F_0_bins = []
    f_bins = xmin*np.logspace(0,10,60,base=2)

    f_0_m = np.empty(len(f_bins))
    f_0_sgm = np.empty(len(f_bins))

    for j in range(0, numinter):
        hS[j] = hS[j][hS[j] > xmin]
        f_0.append(SE.Getf(hS[j], f_bins))
        F_bins, F = SE.GetF(hS[j], xmin, np.max(hS[j]))
        F = np.interp(f_bins, F_bins[0:-1], F)
        F_0.append(F)


    f_0_med = np.median(f_0, axis=0)
    f_0_m = np.mean(f_0, axis=0)
    f_0_sgm = np.sqrt(np.mean((f_0-f_0_m)**2, axis=0))
    f_0_low = np.quantile(f_0, 0.01, axis=0)
    f_0_height = np.quantile(f_0, 0.99, axis=0)

    full_hS = np.concatenate(hS)
    xmax = np.max(full_hS)

    theta = {}
    theta[0] = SE.MLE(full_hS, xmin, max(hS[0])).Lognorm()
    theta[1] = SE.MLE(hS[0], xmin, max(hS[0])).Weibull()
    theta[2] = SE.MLE(hS[0], xmin, max(hS[0])).ParetoMod()

    print("Lognorm", theta[0])
    print("Weibull", theta[1])
    print("ParetoMod", theta[2])

    dist = {}
    dist[0] = SE.lognorm(theta[0][0], theta[0][2])
    dist[1] = SE.weibull(theta[1][0], theta[1][2])
    dist[2] = SE.paretomod(theta[2][0], theta[2][2])


    C = {}
    C[0] = 1 / (dist[0].cdf(xmax) - dist[0].cdf(xmin))
    C[1] = 1 / (dist[0].cdf(xmax) - dist[1].cdf(xmin))
    C[2] = 1 / (dist[0].cdf(xmax) - dist[2].cdf(xmin))

    dist_pdf = {}
    dist_pdf[0] = lambda x : dist[0].pdf(x) * C[0]
    dist_pdf[1] = lambda x : dist[1].pdf(x) * C[1]
    dist_pdf[2] = lambda x : dist[2].pdf(x) * C[2]

    dist_cdf = {}
    dist_cdf[0] = lambda x: ((dist[0].cdf(x) - dist[0].cdf(xmin)) * C[0])
    dist_cdf[1] = lambda x: ((dist[1].cdf(x) - dist[1].cdf(xmin)) * C[1])
    dist_cdf[2] = lambda x: ((dist[2].cdf(x) - dist[2].cdf(xmin)) * C[0])

    f = {}
    f[0] = dist_pdf[0](f_bins)
    f[1] = dist_pdf[1](f_bins)
    f[2] = dist_pdf[2](f_bins)

    F = {}
    F[0] = dist_cdf[0](f_bins)
    F[1] = dist_cdf[1](f_bins)
    F[2] = dist_cdf[2](f_bins)

    Theta["Name"].append(FileName)

    def GetBool(L,H,F):
        return np.sum(((L < F) & (F < H)))/len(F)

    mf = {}
    mf[0] = (F[0][1:] - F[0][0:-1]) / (f_bins[1:] - f_bins[0:-1])
    mf[1] = (F[1][1:] - F[1][0:-1]) / (f_bins[1:] - f_bins[0:-1])
    mf[2] = (F[2][1:] - F[2][0:-1]) / (f_bins[1:] - f_bins[0:-1])

    B = {}
    B[0] = GetBool(f_0_low, f_0_height, mf[0])
    B[1] = GetBool(f_0_low, f_0_height, mf[1])
    index = np.searchsorted(f_bins, theta[2][2])
    if(index < len(mf[2])):
        B[2] = GetBool(f_0_low[index:], f_0_height[index:], mf[2][index:])
    else:
        B[2] = 0

    #%%
    Theta["lognorm"]["boolean"].append(B[0])
    Theta["lognorm"]["sigma"].append(theta[0][0])
    Theta["lognorm"]["lambda"].append(theta[0][2])
    #%%
    Theta["weibull"]["boolean"].append(B[1])
    Theta["weibull"]["alpha"].append(theta[1][0])
    Theta["weibull"]["lambda"].append(theta[1][2])
    #%%
    Theta["paretomod"]["boolean"].append(B[2])
    Theta["paretomod"]["alpha"].append(theta[2][0])
    Theta["paretomod"]["lambda"].append(theta[2][2])

    #%%
    # Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
    # img = cv2.imread(Path_img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # d = 512
    # x0 = 3000
    # y0 = 3500
    # lab_image = cv2.cvtColor(img[y0:y0 + d, x0:x0 + d], cv2.COLOR_RGB2Lab)
    # lab_image, _, _ = cv2.split(lab_image)
    # font = font_manager.FontProperties(fname="/home/chinkin/Times_New_Roman.ttf",
    #                        style='normal', size=16)
    #
    # fig = plt.figure(figsize=(10, 5))
    # ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    # ax[0].imshow(cv2.merge((lab_image, lab_image, lab_image)))
    # #
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')
    # H = np.insert(f_0_height, 0, f_0_height[0])
    # L = np.append(f_0_low, f_0_low[-1])
    # ax[1].fill_between(f_bins, L, H,
    #                    alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
    # ax[1].plot(f_bins, f[0],
    #            color='black',label="Lognormal")
    # ax[1].plot(f_bins, f[1],
    #        color='black',linestyle='dashdot',label="Exponencial")
    # ax[1].plot(f_bins[f_bins > theta[2][2]], f[2][f_bins > theta[2][2]],
    #        color='black',linestyle='dashed',label="Power (Pareto)")
    # ax[1].set_ylim((f_0_low[-1], f_0_height[0]))
    # ax[1].set_xlim((f_bins[0], f_bins[-2]))
    # ax[1].legend(loc='lower left')
    # ax[1].set_xlabel("S, num pixel")
    # ax[1].set_ylabel("Probobility density")
    # fig.suptitle(FileName, fontsize=16)
    # #fig.savefig("temp/StatPictures/" + FileName + "_pf_S.png")
    # plt.show()
    # plt.close("all")
    # exit()



header = ["Номер Образца",
          "Lognorm boolean",
          "Lognorm sigma",
          "Lognorm lambda",
          "Weibull boolean",
          "Weibull alpha",
          "Weibull lambda",
          "Pareto boolean",
          "Pareto alpha",
          "Pareto lambda"]


with open('temp/StatisticResult.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    for i in range(0,len(Theta["Lognorm"]["s"])):
        data = [Theta["Name"][i],
                round(Theta["lognorm"]["boolean"][i], 2),
                round(Theta["lognorm"]["sigma"][i], 2),
                round(Theta["lognorm"]["lambda"][i], 2),
                round(Theta["weibull"]["boolean"][i], 2),
                round(Theta["weibull"]["alpha"][i], 2),
                round(Theta["weibull"]["lambda"][i], 2),
                round(Theta["Pareto"]["boolean"][i], 2),
                round(Theta["pareto"]["alpha"][i], 2),
                round(Theta["pareto"]["lambda"][i], 2)]
        # write the data
        writer.writerow(data)

exit()