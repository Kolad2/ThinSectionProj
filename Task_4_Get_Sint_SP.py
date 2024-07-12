import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import time
from ThinSS import ThinSS
from Shpreader import get_shp_poly
import random
import math

Path0 = "/media/kolad/HardDisk/ThinSection"
StatisticPath = "temp/"
FileNames = os.listdir(Path0)
intermax = 500


if not os.path.exists(StatisticPath):
	print("Путь статистики не существует: " + StatisticPath)
	exit()
else:
	if not os.path.exists(StatisticPath + "StatisticSintData"):
		os.mkdir(StatisticPath + "StatisticSintData")


for FileName in FileNames:
	print("Start", FileName)
	Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
	Path_rsf = Path_dir + "RSF_edges" + "/" + FileName + "_edges.tif"
	Path_shape = Path_dir + "Joined/" + FileName + "_joined"
	Path_edges = Path_dir + "Shape/" + FileName
	if not os.path.exists(Path_img):
		print("Файл изображения не найден")
		continue
	if not os.path.exists(Path_rsf):
		print("Не найдена граница RSF")
		continue
	if not os.path.exists(Path_shape + ".shp"):
		print("Шейп файл не найден")
		continue
	if not os.path.exists(Path_edges + ".shp"):
		print("Шейп файл границ не найден")
		continue
	if os.path.exists("temp/StatisticSintData/" + FileName + "/" + FileName + "_S.mat"):
		print("Найдена копия мат файла, пропуск")
		continue
	# image loading
	img = cv2.imread(Path_img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# RSF_result load
	img_rsf = cv2.imread(Path_rsf)
	result_rsf, _, _ = cv2.split(img_rsf)
	# Shape-file and lineaments load
	poly = get_shp_poly(Path_shape)
	result_line = np.zeros(img.shape[0:2], dtype=np.uint8)
	result_line = cv2.polylines(result_line, poly, False, 255, 3)
	# edge shape-file loading
	edge_poly = get_shp_poly(Path_edges)
	print("Segmentation start", FileName)
	S = []
	P = []
	for i in range(0, intermax, 1):
		print(FileName, i, "/", intermax, flush=True)
		polys = get_shp_poly(Path_shape)
		result_line = np.zeros(img.shape[0:2], dtype=np.uint8)
		polys2 = []
		xi = 0.3 * math.sqrt(random.uniform(0, 1))
		for poly in polys:
			if random.uniform(0, 1) > xi:
				polys2.append(poly)
		result_line = cv2.polylines(result_line, polys2, False, 255, 3)
		TS = ThinSS(img, result_rsf, result_line)
		edge_mask = TS.get_edge(edge_poly)
		TS.RunSegmentation(edge_mask)
		lS, lP = TS.get_SP()
		S.append(lS)
		P.append(lP)
	dict = {'S': S, 'P': P}
	if not os.path.exists(StatisticPath + "StatisticSintData/" + FileName):
		os.mkdir(StatisticPath + "StatisticSintData/" + FileName)
	scipy.io.savemat(StatisticPath + "StatisticSintData/" + FileName + "/" + FileName + "_S.mat", dict)
