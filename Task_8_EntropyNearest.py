import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors
import random
import csv
import StatisticEstimation as SE

def GetH(P):
	return -np.sum(P * np.log2(P, out=np.zeros_like(P), where=(P != 0)), axis=None)


def Getdphi(phi1: np.ndarray, phi2: np.ndarray):
	dphi = np.abs(phi1 - phi2)
	return np.min([dphi, 180 - dphi], axis=0)


def converter(FileName):
	FileName = FileName.replace("B", "Б-")
	g = FileName.find("a")
	if g != -1:
		return FileName[0:g]
	g = FileName.find("b")
	if g != -1:
		return FileName[0:g]
	return FileName


bins_phi = np.linspace(0, 180, 120)

Path_Names = "/media/kolad/HardDisk/ThinSection/"
FileNames = os.listdir(Path_Names)

Path0 = "/media/kolad/HardDisk/StatisticData/StatMatrData/"

with open('/media/kolad/HardDisk/StatisticData/Petrograph.csv', newline='') as csvfile1:
	rows_petro = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
	tb_petro = pd.DataFrame(rows_petro[1:], columns=rows_petro[0])
	del rows_petro

N = len(FileNames)

TSI = {
	"Номер Образца": FileNames,
	"H_drdphi": np.empty(N, np.float16),
	"H_dr": np.empty(N, np.float16),
	"H_r": np.empty(N, np.float16),
	"H_dphi": np.empty(N, np.float16),
	"H_phi": np.empty(N, np.float16),
	"c_med": np.empty(N, np.float16),
	"dr_med": np.empty(N, np.float16),
	"smu": np.empty(N, np.float16),
	"s": np.empty(N, np.float16),
	"ПетрографТипы": [None]*N,
	"ТипыТектонитов": [None]*N,
	"%матрикса": [None]*N
}

for i, FileName in enumerate(FileNames):
	FilePath = Path0 + FileName + "_J.mat"
	print(FileName, "(",  i, "/", N, ")")
	dict = scipy.io.loadmat(FilePath, squeeze_me=True)
	dict["phi"][dict["phi"] < 0] = 180 + dict["phi"][dict["phi"] < 0]
	array = (dict["S"] > 20)
	dict["phi"] = np.array(dict["phi"][array])
	dict["xC"] = np.array(dict["xC"][array])
	dict["yC"] = np.array(dict["yC"][array])
	dict["a"] = np.array(dict["a"][array])
	dict["b"] = np.array(dict["b"][array])
	dict["S"] = np.array(dict["S"][array])

	PetroName = converter(TSI["Номер Образца"][i])
	row = tb_petro[tb_petro["НомерОбразца"].isin([PetroName])]
	TSI["ПетрографТипы"][i] = (row["ПетрографТипы"].values.tolist())[0]
	TSI["ТипыТектонитов"][i] = (row["ТипыТектонитов"].values.tolist())[0]
	TSI["%матрикса"][i] = (row["%матрикса"].values.tolist())[0]

	c = dict["a"] / dict["b"]

	bins_c = np.linspace(1, 5, 60)
	f_c, _ = np.histogram(c, bins=bins_c, density=True)
	TSI["c_med"][i] = np.median(c)

	f_phi, _ = np.histogram(dict["phi"], bins=bins_phi, density=True)

	theta = SE.GetThetaLognorm(dict["S"], 20, 10**10)
	TSI["s"][i] = theta[0]
	TSI["smu"][i] = theta[2]
	# ====
	X = np.column_stack([dict["xC"], dict["yC"]])
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
	dr, ind = nbrs.kneighbors(X)
	dr = dr[:, 1]
	ind = ind[:, 1]
	dphi = Getdphi(dict["phi"], dict["phi"][ind])

	TSI["dr_med"][i] = np.median(dr)
	TSI["H_r"][i] = np.mean(np.log(dr))

	# =====

	bins_dr = np.linspace(0, 80, 120)
	f_dr, _ = np.histogram(dr, bins_dr, density=True)

	bins_dphi = np.linspace(0, 90, 120)
	f_dphi, _ = np.histogram(dphi, bins_dphi, density=True)

	f_drdphi, _, _ = np.histogram2d(dr, dphi, bins=(bins_dr, bins_dphi), density=True)

	# =====

	TSI["H_drdphi"][i] = GetH(f_drdphi * bins_dr[1] * bins_dphi[1])
	TSI["H_dr"][i] = GetH(f_dr * bins_dr[1])
	TSI["H_dphi"][i] = GetH(f_dphi * bins_dphi[1])
	TSI["H_phi"][i] = GetH(f_phi * bins_phi[1])



header = TSI.keys()

print("Запись файла")
with open('temp/StatisticResult.csv', 'w', encoding='UTF8', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(header)
	for i in range(0, N):
		data = []
		for key in header:
			data.append(TSI[key][i])
		print(data[0])
		writer.writerow(data)
