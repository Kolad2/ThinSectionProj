import os
import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle

Path0 = "/media/kolad/HardDisk/ThinSection"
FileNames = os.listdir(Path0)
print(FileNames)


for FileName in FileNames:
    print(FileName)
    Path_dir = Path0 + "/" + FileName + "/"
    Path_shape = Path_dir + "Joined/" + FileName + "_joined"
    Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
    Path_lin = Path_dir + "Lineaments/" + FileName + "_lin.tif"
    if not os.path.exists(Path_shape + ".shp"):
        print("Шейп файл не найден")
        continue
    if not os.path.exists(Path_img):
        print("Файл изображения не найден")
        continue
    if os.path.exists(Path_dir + "Lineaments"):
        if os.path.exists(Path_lin):
            print("Файл фото линеаментов сущестует")
            continue
    else:
        os.mkdir(Path_dir + "Lineaments")

    poly = get_shp_poly(Path_shape)
    img = cv2.imread(Path_img)
    sh = img.shape
    img_lines = np.zeros(sh[0:2], dtype=np.uint8)
    img_lines = cv2.polylines(img_lines, poly, False, 255, 3)
    cv2.imwrite(Path_lin, img_lines)