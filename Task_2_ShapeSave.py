import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
import scipy
import time
import pickle
import math
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSS import ThinSS
from scipy.spatial import cKDTree, KDTree




Path0 = "/media/kolad/HardDisk/ThinSection"
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
	n = 10
	edge_mask = 255 - np.zeros((2 ** n, 2 ** n), np.uint8)
	edge_zero = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
	cv2.fillPoly(edge_mask, pts=[edge_zero, edge_poly[0]], color=0)

	TS = ThinSS(img, result_rsf, result_line)
	edge_mask = TS.get_edge(edge_poly)
	TS.RunSegmentation(edge_mask)

	TS.area_bg = TS.area_bg[1000:1000+2 ** n, 1000:1000+2 ** n]
	TS.area_marks = TS.area_marks[1000:1000+2 ** n, 1000:1000+2 ** n]
	TS.img = TS.img[1000:1000 + 2 ** n, 1000:1000 + 2 ** n]

	print("Segmentation end", FileName)
	print("Segmentation save start", FileName)
	mask_write_treads("Shapes/Shape_" + FileName + "/Shape_" + FileName + "/Shape_" + FileName, TS.get_masks())
	cv2.imwrite("Shapes/Shape_" + FileName + "/img_" + FileName + ".tiff", TS.img)

