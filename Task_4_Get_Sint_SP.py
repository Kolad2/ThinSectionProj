import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import time as t
from ThinSS import ThinSS
from Shpreader import get_shp_poly
import random
import math
import multiprocessing as mp

Path0 = "/media/kolad/HardDisk/ThinSection"
StatisticPath = "/media/kolad/HardDisk/StatisticData/"
FileNames = os.listdir(Path0)
intermax = 500


if not os.path.exists(StatisticPath):
	print("Путь статистики не существует: " + StatisticPath)
	exit()
else:
	if not os.path.exists(StatisticPath + "StatisticSintData"):
		os.mkdir(StatisticPath + "StatisticSintData")


def GetParams(img, polys, edge_poly, result_rsf):
	result_line = np.zeros(img.shape[0:2], dtype=np.uint8)
	loc_polys = []
	xi = 0.3*math.sqrt(random.uniform(0, 1))
	for poly in polys:
		if random.uniform(0, 1) > xi:
			loc_polys.append(poly)
	result_line = cv2.polylines(result_line, loc_polys, False, 255, 3)
	TS = ThinSS(img, result_rsf, result_line)
	edge_mask = TS.get_edge(edge_poly)
	TS.RunSegmentation(edge_mask)
	S, P = TS.get_SP()
	return S, P

def worker(S, P, img, polys, edge_poly, result_rsf):
	lS, lP = GetParams(img, polys, edge_poly, result_rsf)
	S.append(lS)
	P.append(lP)

def ImageProcessing(FileName):
	Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
	Path_rsf = Path_dir + "RSF_edges" + "/" + FileName + "_edges.tif"
	Path_shape = Path_dir + "Joined/" + FileName + "_joined"
	Path_edges = Path_dir + "Shape/" + FileName
	if not os.path.exists(Path_img):
		print("Файл изображения не найден")
		return
	if not os.path.exists(Path_rsf):
		print("Не найдена граница RSF")
		return
	if not os.path.exists(Path_shape + ".shp"):
		print("Шейп файл не найден")
		return
	if not os.path.exists(Path_edges + ".shp"):
		print("Шейп файл границ не найден")
		return
	if os.path.exists(StatisticPath + "StatisticSintData/" + FileName + "/" + FileName + "_S.mat"):
		print("Найдена копия мат файла, пропуск")
		return
	# image loading
	img = cv2.imread(Path_img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# RSF_result load
	img_rsf = cv2.imread(Path_rsf)
	result_rsf, _, _ = cv2.split(img_rsf)
	# edge shape-file loading
	edge_poly = get_shp_poly(Path_edges)
	print("Segmentation start", FileName)
	polys = get_shp_poly(Path_shape)

	with (mp.Manager() as manager):
		# Создание общего списка
		S = manager.list()
		P = manager.list()
		nw = 4
		for i in range(0, 128, 1):
			p = {}
			print(i)
			#print(t.time())
			for j in range(0, nw, 1):
				p[j] = mp.Process(target=worker, args=(S, P, img, polys, edge_poly, result_rsf))
				p[j].start()
			for j in range(0, nw, 1):
				p[j].join()
			#print(t.time())
		S = list(S)
		P = list(P)
	dict = {'S': S, 'P': P}
	if not os.path.exists(StatisticPath + "StatisticSintData/" + FileName):
		os.mkdir(StatisticPath + "StatisticSintData/" + FileName)
	scipy.io.savemat(StatisticPath + "StatisticSintData/" + FileName + "/" + FileName + "_S.mat", dict)

for FileName in FileNames:
	print("Start", FileName)
	ImageProcessing(FileName)

