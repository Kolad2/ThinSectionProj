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
import matplotlib as mpl
import matplotlib.patheffects as path_effects
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
    "paretomod": {"boolean": [], "alpha": [], "lambda": []},
    "gengamma": {"boolean": [], "a": [], "b": [], "lambda": []}}

for i in range(0, len(tb_stat["Номер Образца"])):
    h = tb_stat["Номер Образца"].values[i]
    #print(h)


xmin = 20
f_bins = xmin*np.logspace(0,10,180,base=2)
xmax = f_bins[-1]
numinter = 500


for i0, FileName in enumerate(FileNames):
    print(FileName,"(", i0+1 , "/", len(FileNames), ")")
    # if len(tb_stat[tb_stat["Номер Образца"] == FileName]) > 0:
    #     print("Статистика образца найдена, пропуск")
    #     continue

    hS, P = GetData(FileName)

    Path_dir = Path0 + "/" + FileName + "/"
    F_0 = {}
    f_0 = {}
    F_0_bins = []


    f_0_m = np.empty(len(f_bins))
    f_0_sgm = np.empty(len(f_bins))

    for j in range(0, numinter):
        hS[j] = hS[j][(hS[j] >= xmin) & (hS[j] <= xmax)]
        f_0[j] = (SE.Getf(hS[j], f_bins))
        X, F = SE.get_ecdf(hS[j], xmin)
        F_0[j] = np.interp(f_bins, X, F)

    Ns = np.mean([1/math.sqrt(len(x)) for x in hS])
    Ns0 = np.mean([len(x) for x in hS])
    dst = (sp.stats.kstwobign.ppf(0.95)/math.sqrt(Ns0))

    F_0 = list(F_0.values())
    f_0 = list(f_0.values())

    f_0_med = np.median(f_0, axis=0)
    f_0_m = np.mean(f_0, axis=0)
    f_0_sgm = np.sqrt(np.mean((f_0-f_0_m)**2, axis=0))
    f_0_low = np.quantile(f_0, 0.05, axis=0)
    f_0_height = np.quantile(f_0, 0.95, axis=0)

    F_0_low = np.quantile(F_0, 0.05, axis=0) - dst
    F_0_m = np.quantile(F_0, 0.5, axis=0)
    F_0_height = np.quantile(F_0, 0.95, axis=0) + dst

    full_hS = np.concatenate(hS)
    xmax = np.max(full_hS)

    theta = {}
    theta[0] = SE.MLE(full_hS, xmin, xmax).Lognorm()
    theta[1] = SE.MLE(hS[0], xmin, xmax).Weibull()
    theta[2] = SE.MLE(hS[0], xmin, xmax).ParetoMod()
    theta[3] = SE.MLE(hS[0], xmin, xmax).Gengamma()

    #exit()
    # print("Lognorm", theta[0])
    # print("Weibull", theta[1])
    # print("ParetoMod", theta[2])
    # print("Gengamma", theta[3])

    dist = {}
    dist[0] = SE.lognorm(theta[0][0], theta[0][2])
    dist[1] = SE.weibull(theta[1][0], theta[1][2])
    dist[2] = SE.paretomod(theta[2][0], theta[2][2])
    dist[3] = SE.gengamma(theta[3][0], theta[3][1], theta[3][3])

    def GetBool(L,H,F):
        return np.sum(((L <= F) & (F <= H)))/len(F)

    C = {}
    dist_pdf = {}
    dist_cdf = {}
    f = {}
    F = {}
    B = {}
    for i in range(0, 4):
        C[i] = 1 / (dist[i].cdf(xmax) - dist[i].cdf(xmin))
        dist_pdf[i] = lambda x: dist[i].pdf(x) * C[i]
        dist_cdf[i] = lambda x: ((dist[i].cdf(x) - dist[i].cdf(xmin)) * C[i])
        f[i] = dist_pdf[i](f_bins)
        F[i] = dist_cdf[i](f_bins)

    B[0] = GetBool(F_0_low, F_0_height, F[0])
    B[1] = GetBool(F_0_low, F_0_height, F[1])
    index = np.searchsorted(f_bins, theta[2][2], side = 'right')
    B[2] = GetBool(F_0_low[index:], F_0_height[index:], F[2][index:])
    B[3] = GetBool(F_0_low, F_0_height, F[3])

    Theta["Name"].append(FileName)
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
    Theta["gengamma"]["boolean"].append(B[3])
    Theta["gengamma"]["a"].append(theta[3][0])
    Theta["gengamma"]["b"].append(theta[3][1])
    Theta["gengamma"]["lambda"].append(theta[3][3])
    print(B)
    #continue
    #%%
    Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
    img = cv2.imread(Path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = 512
    x0 = 3000
    y0 = 3500
    #%%
    font = mpl.font_manager.FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf",
                           style='normal', size=16)

    dictpics = sp.io.loadmat(StatisticPath + "StatisticSintData/" + FileName + "/" + FileName + "_pics.mat", squeeze_me=True)
    #%%
    fig = plt.figure(figsize=(16, 4))
    ax = [fig.add_subplot(1, 4, 1),
          fig.add_subplot(1, 4, 2),
          fig.add_subplot(1, 4, 3),
          fig.add_subplot(1, 4, 4)]

    #%%
    lab_image = cv2.cvtColor(img[y0:y0 + d, x0:x0 + d], cv2.COLOR_RGB2Lab)
    lab_image, _, _ = cv2.split(lab_image)
    ax[0].imshow(dictpics["PIC"], origin='lower')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].imshow(dictpics["SEG"], origin='lower')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    #%%
    v = 0
    ax[2].fill_between(f_bins*(0.25**2), F_0_low - F_0_m*v,  F_0_height - F_0_m*v,color='grey',label='1')
    #ax[1].plot(f_bins*(0.25**2), F[0] - F_0_m*v, color='black',label='2')
    #ax[1].plot(f_bins*(0.25**2), F[1] - F_0_m*v, color='black',linestyle='--',label='3')
    #ax[1].plot(f_bins*(0.25**2), F[2] - F_0_m*v, color='black',linestyle='-.',label='4')
    #ax[1].plot(f_bins*(0.25**2), F[3] - F_0_m*1, color='green')
    ax[2].set_xscale('log')
    #ax[2].legend(prop=font)
    #ax[1].set_ylim((F_0_low[-1], F_0_height[0]))
    ax[2].set_ylim((0, 1))
    ax[2].set_xlim((f_bins[0]*(0.25**2), f_bins[-1]*(0.25**2)))
    for label in ax[2].get_yaxis().get_ticklabels():
        label.set_fontproperties(font)
    for label in ax[2].get_xaxis().get_ticklabels():
        label.set_fontproperties(font)
    ax[2].set_xlabel("s, мкм$^2$", fontproperties=font)
    ax[2].set_ylabel(r"F, усл.ед.", fontproperties=font)
    # %%
    v = 1
    ax[3].fill_between(f_bins * (0.25 ** 2), F_0_low - F_0_m * v, F_0_height - F_0_m * v, color='grey', label='1')
    ax[3].plot(f_bins*(0.25**2), F[0] - F_0_m*v, color='black',label='2')
    ax[3].plot(f_bins*(0.25**2), F[1] - F_0_m*v, color='black',linestyle='--',label='3')
    ax[3].plot(f_bins*(0.25**2), F[2] - F_0_m*v, color='black',linestyle='-.',label='4')
    #ax[3].plot(f_bins*(0.25**2), F[3] - F_0_m*v, color='green')
    ax[3].set_xscale('log')
    #ax[3].set_ylim((0, 1))

    ax[3].set_xlim((f_bins[0] * (0.25 ** 2), f_bins[-1] * (0.25 ** 2)))
    for label in ax[3].get_yaxis().get_ticklabels():
        label.set_fontproperties(font)
    for label in ax[3].get_xaxis().get_ticklabels():
        label.set_fontproperties(font)
    ax[3].set_xlabel("s, мкм$^2$",fontproperties=font)
    ax[3].set_ylabel(r"F-F$_{\text{m}}$, усл.ед.", fontproperties=font)
    ax[3].legend(prop=font, loc='upper right', ncol=2)
    # %%
    # Координаты начала и конца масштабной линейки
    def plotScaleLine(ax, start, end):
        dx = (start[0] - end[0])
        dy = (start[1] - end[1])
        ca = dx/math.sqrt(dx*dx + dy*dy)
        sa = dy/math.sqrt(dx*dx + dy*dy)
        ax.plot([start[0], end[0]], [start[1], end[1]], transform=ax.transAxes, color='white', lw=5)
        ax.plot(
            [start[0] + 0.01*sa, start[0] - 0.01*sa],
            [start[1] - 0.01*ca, start[1] + 0.01*ca], transform=ax.transAxes, color='white', lw=5)
        ax.plot(
            [end[0] - 0.01*sa, end[0] + 0.01*sa],
            [end[1] + 0.01*ca, end[1] - 0.01*ca], transform=ax.transAxes, color='white', lw=5)
        #
        ax.plot([start[0], end[0]], [start[1], end[1]], transform=ax.transAxes, color='black', lw=2)
        ax.plot(
            [start[0]-0.01*sa, start[0]+0.01*sa],
            [start[1]+0.01*ca, start[1]-0.01*ca], transform=ax.transAxes, color='black', lw=2)
        ax.plot(
            [end[0] - 0.01*sa, end[0] + 0.01*sa],
            [end[1] + 0.01*ca, end[1] - 0.01*ca], transform=ax.transAxes, color='black', lw=2)


    start = [0.79, 0.1]
    end = [0.97, 0.1]
    plotScaleLine(ax[0], start, end)
    plotScaleLine(ax[1], start, end)
    start = ax[0].transAxes.transform(start) - [-60, 15]
    end = ax[0].transAxes.transform(end) - [-60, 15]
    print(start)
    text = ax[0].annotate('25мкм', xy=(450, 75), ha='center', color='black', fontproperties=font)
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),path_effects.Normal()])
    text = ax[1].annotate('25мкм', xy=(450, 75), ha='center', color='black', fontproperties=font)
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
    #
    fig.tight_layout()
    fig.savefig("temp/StatPictures/" + FileName + "_pf_S.png")
    #plt.show()
    plt.close("all")

    continue
    #exit()
    #%%
    Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
    img = cv2.imread(Path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = 512
    x0 = 3000
    y0 = 3500
    lab_image = cv2.cvtColor(img[y0:y0 + d, x0:x0 + d], cv2.COLOR_RGB2Lab)
    lab_image, _, _ = cv2.split(lab_image)
    # font = font_manager.FontProperties(fname="/home/chinkin/Times_New_Roman.ttf",
    #                        style='normal', size=16)
    fig = plt.figure(figsize=(10, 5))
    ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    ax[0].imshow(cv2.merge((lab_image, lab_image, lab_image)))
    #
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    H = np.insert(f_0_height, 0, f_0_height[0])
    L = np.append(f_0_low, f_0_low[-1])
    ax[1].fill_between(f_bins, L, H,
                       alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
    ax[1].plot(f_bins, f[0],
               color='black',label="Lognormal")
    ax[1].plot(f_bins, f[1],
           color='black',linestyle='dashdot',label="Exponencial")
    ax[1].plot(f_bins[f_bins > theta[2][2]], f[2][f_bins > theta[2][2]],
           color='black',linestyle='dashed',label="Power (Pareto)")
    ax[1].set_ylim((f_0_low[-1], f_0_height[0]))
    ax[1].set_xlim((f_bins[0], f_bins[-2]))
    ax[1].legend(loc='lower left')
    ax[1].set_xlabel("S, num pixel")
    ax[1].set_ylabel("Probobility density")
    fig.suptitle(FileName, fontsize=16)
    #fig.savefig("temp/StatPictures/" + FileName + "_pf_S.png")
    plt.show()
    plt.close("all")
    exit()



header = ["Номер Образца",
          "Lognorm boolean",
          "Lognorm sigma",
          "Lognorm lambda",
          "Weibull boolean",
          "Weibull alpha",
          "Weibull lambda",
          "Pareto boolean",
          "Pareto alpha",
          "Pareto lambda",
          "Gengamma boolean",
          "Gengamma a",
          "Gengamma b",
          "Gengamma lambda",
          ]


with open('temp/StatisticResult.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    for i in range(0,len(Theta["Name"])):
        data = [Theta["Name"][i],
                round(Theta["lognorm"]["boolean"][i], 2),
                round(Theta["lognorm"]["sigma"][i], 2),
                round(Theta["lognorm"]["lambda"][i], 2),
                round(Theta["weibull"]["boolean"][i], 2),
                round(Theta["weibull"]["alpha"][i], 2),
                round(Theta["weibull"]["lambda"][i], 2),
                round(Theta["paretomod"]["boolean"][i], 2),
                round(Theta["paretomod"]["alpha"][i], 2),
                round(Theta["paretomod"]["lambda"][i], 2),
                round(Theta["gengamma"]["boolean"][i], 2),
                round(Theta["gengamma"]["a"][i], 2),
                round(Theta["gengamma"]["b"][i], 2),
                round(Theta["gengamma"]["lambda"][i], 2)]
        # write the data
        writer.writerow(data)

exit()