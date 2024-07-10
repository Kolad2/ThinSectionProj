import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shapefile

def get_shp_poly(path):
    shapes: list[Any]
    with shapefile.Reader(path) as shp:
        shapes = shp.shapes()
        bbox = shp.bbox
    poly = []
    for i in range(len(shapes)):
        poly.append(np.array(shapes[i].points, np.int32) * [1, -1])
    return poly
