import math
import os

import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle
from rsf_edges import modelini, get_model_edges, modelgpu
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation


class RSF_overcrop:
	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.img[y:y + dy, x:x + dx]
		self.result_rsf[y+ddy:y+dy-ddy, x+ddx:x+dx-ddx] = self.model.get_model_edges(img_small)[ddy:dy-ddy, ddx:dx-ddx]

	def get_full_edge(self, dx, dy, ddx, ddy):
		jmax = math.floor((self.sh[0] - 2 * ddy) / (dy - 2 * ddy))
		imax = math.floor((self.sh[1] - 2 * ddx) / (dx - 2 * ddx))
		for i in range(0, imax):
			for j in range(0, jmax):
				if (i*jmax + j) % 50 == 0:
					print(i*jmax + j,"/",imax*jmax)
				self.get_crop_edge((dx - 2 * ddx) * i, (dy - 2 * ddy) * j, dx, dy, ddx, ddy)
			y = (dy - 2 * ddy) * jmax
			self.get_crop_edge((dx - 2 * ddx) * i, y, dx, self.sh[0] - y, ddx, ddy)
		for j in range(0, jmax):
			x = (dx - 2 * ddx) * imax
			self.get_crop_edge(x, (dy - 2 * ddy) * j, self.sh[1] - x, dy, ddx, ddy)

	def __init__(self,img):
		self.img = img
		self.sh = img.shape
		self.model = modelgpu()
		self.result_rsf = np.zeros(img.shape[0:2], np.float32)


dx = 1000
dy = 1000
ddx = 256
ddy = 256
Path0 = "/media/kolad/HardDisk/TSnew"
#Path0 = "/media/kolad/HardDisk/ThinSection"

FileNames = os.listdir(Path0)
print(FileNames)

for FileName in FileNames:
	print(FileName)
	Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
	Path_edges_img = Path_dir + "RSF_edges" + "/" + FileName + "_edges.tif"
	if not os.path.exists(Path_dir + "RSF_edges"):
		os.mkdir(Path_dir + "RSF_edges")
	if os.path.exists(Path_edges_img):
		print("Файл границ " + FileName + " существует, пропуск")
		continue
	if not os.path.exists(Path_img):
		print("Файл " + FileName + " не найден")
		exit()
		continue
	img = cv2.imread(Path_img)
	RSF = RSF_overcrop(img)
	RSF.get_full_edge(dx, dy, ddx, ddy)
	result_rsf = np.uint8((RSF.result_rsf / RSF.result_rsf.max()) * 255)
	img_rsf = cv2.merge((result_rsf, result_rsf, result_rsf))
	cv2.imwrite(Path_edges_img, img_rsf)
