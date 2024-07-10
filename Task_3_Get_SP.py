import os
import sys
import PathCreator
from typing import Any
import numpy as np
import cv2
from Shpreader import get_shp_poly
import matplotlib.pyplot as plt
from ThinSS import ThinSS
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerStepPatch

Path0 = "/media/kolad/HardDisk/ThinSection"
PathExit = "/media/kolad/HardDisk/StatisticData/PicturesSegment/"
FileNames = os.listdir(Path0)

for FileName in FileNames:
	FileName = "B21-51b"
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

	TS = ThinSS(img, result_rsf, result_line)
	edge_mask = TS.get_edge(edge_poly)
	TS.RunSegmentation(edge_mask)

	pic = [None, None, None, None, None]
	d = 512
	x0 = 3500
	y0 = 4500
	lab_image = cv2.cvtColor(img[y0:y0 + d, x0:x0 + d], cv2.COLOR_RGB2Lab)
	pic[0], _, _ = cv2.split(lab_image)
	pic[1] = result_line[y0:y0 + d, x0:x0 + d]
	_, result_rsf = cv2.threshold(result_rsf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	pic[2] = result_rsf[y0:y0 + d, x0:x0 + d]
	#pic[3] = 255 - TS.area_bg[y0:y0 + d, x0:x0 + d]

	pic[3] = TS.temp_area_marks[y0:y0 + d, x0:x0 + d]
	pic[3][pic[3] == -1] = 0
	for i in np.unique(pic[0])[1:]:
		pic[3][pic[3] == i] = np.random.randint(50, 254)
	pic[3] = pic[3].astype(np.uint8)

	pic[4] = TS.area_marks[y0:y0 + d, x0:x0 + d]
	pic[4][pic[4] == -1] = 0
	for i in np.unique(pic[3])[1:]:
		pic[4][pic[4] == i] = np.random.randint(50, 254)
	pic[4] = pic[4].astype(np.uint8)



	Y, X = np.indices((d, d))

	fig = plt.figure(figsize=(14, 9))
	fig.suptitle(FileName, fontsize=16)
	ax = [fig.add_subplot(2, 3, 1),
	      fig.add_subplot(2, 3, 2),
	      fig.add_subplot(2, 3, 3),
	      fig.add_subplot(2, 3, 4),
	      fig.add_subplot(2, 3, 5)]
	chars = ["a","c","d","b","e"]

	patch = mpl.lines.Line2D([0], [0], color='black', label='0.1 мм', linestyle='solid',
	                       linewidth=1, marker='|')
	for i in range(0, 5):
		ax[i].pcolor(X*0.25, Y*0.25, cv2.merge((pic[i], pic[i], pic[i])))
		ax[i].set_xlabel(chars[i] +")")
		ax[i].legend(
			handles=[patch],
		    handlelength=2.0,
		    handleheight=1.5,
		    framealpha=1,
			prop=font)
		ax[i].get_xaxis().set_visible(False)
		ax[i].get_yaxis().set_visible(False)

	fig.savefig(PathExit + FileName + "_pf_S.png")
	plt.show()
	plt.close("all")
	exit()
	#S, P = TS.get_SP()
	#dict = {'S': S, 'P': P}
	#scipy.io.savemat("temp/StatisticRawData/" + FileName + "_S.mat", dict)