import math
import os
import sys
import PathCreator
from typing import Any
import numpy as np
import cv2
from Shpreader import get_shp_poly
import matplotlib.pyplot as plt
from ThinSS import ThinSS
import scipy

pathsave = "/media/kolad/HardDisk/StatisticData/StatMatrData/"
Path0 = "/media/kolad/HardDisk/ThinSection"
PathExit = "temp/PicturesSegment/"
FileNames = os.listdir(Path0)

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
	if os.path.exists(pathsave + FileName + "_J.mat"):
		print("Файл статистики найден, пропуск")
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


	TS = ThinSS(img, result_rsf, result_line)
	edge_mask = TS.get_edge(edge_poly)
	print("Segmentation start", FileName)
	TS.RunSegmentation(edge_mask)
	print("Segmentation end", FileName)
	#S, P = TS.get_SP()
	res = TS.get_momentum()
	scipy.io.savemat(pathsave + FileName + "_J.mat", res)

	# d = 512
	# x0 = 3000
	# y0 = 3500
	# pic = TS.area_marks[y0:y0 + d, x0:x0 + d]
	# pic[pic == -1] = 0
	# u = np.unique(pic)[1:]
	# for i in u:
	# 	pic[pic == i] = np.random.randint(50, 254)
	# pic = pic.astype(np.uint8)
	# pic = cv2.merge((pic, pic, pic))
	# for i in range(0,len(res["xC"])):
	# 	exC = np.floor(res["xC"][i]).astype(np.int32) - x0
	# 	eyC = np.floor(res["yC"][i]).astype(np.int32) - y0
	# 	ea = np.floor(res["a"][i]).astype(np.int32)
	# 	eb = np.floor(res["b"][i]).astype(np.int32)
	# 	cv2.ellipse(pic, (exC, eyC), (ea, eb), res["phi"][i], 0, 360, (0,0,255), 1)
	# fig = plt.figure(figsize=(10, 10))
	# fig.suptitle(FileName, fontsize=16)
	# ax = [fig.add_subplot(1, 1, 1)]
	# ax[0].imshow(pic)
	# plt.show()
	# plt.close("all")
	#S, P = TS.get_SP()
	#dict = {'S': S, 'P': P}
