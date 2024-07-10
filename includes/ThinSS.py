import numpy as np
import cv2
import time
import math
from random import shuffle
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory
from time import time as t

def GetElementSPJ(col, row, ccnts, i, res):

    COL = col[ccnts[i]:ccnts[i + 1]]
    ROW = row[ccnts[i]:ccnts[i + 1]]
    # Xmin, Xmax = np.amin(COL), np.amax(COL) + 1
    # Ymin, Ymax = np.amin(ROW), np.amax(ROW) + 1
    # mask = np.zeros((Ymax - Ymin + 2, Xmax - Xmin + 2), 'uint8')
    # mask[ROW-Ymin+1, COL-Xmin+1] = 1
    i = i - 3
    res["S"][i] = len(COL)
    res["xC"][i] = sum(COL) / res["S"][i]
    res["yC"][i] = sum(ROW) / res["S"][i]
    res["Jxx"][i] = sum(1 / 12 + (COL - res["xC"][i]) ** 2) / res["S"][i]
    res["Jyy"][i] = sum(1 / 12 + (ROW - res["yC"][i]) ** 2) / res["S"][i]
    res["Jxy"][i] = sum((ROW - res["yC"][i]) * (COL - res["xC"][i])) / res["S"][i]
    A = np.array([[res["Jxx"][i], res["Jxy"][i]],
                  [res["Jxy"][i], res["Jyy"][i]]])
    e, v = np.linalg.eig(A)
    ei = np.argmax(e)
    res["phi"][i] = np.degrees(np.arctan2(v[1, ei], v[0, ei]))
    if (ei == 0):
        res["a"][i] = math.sqrt(4 * e[0])
        res["b"][i] = math.sqrt(4 * e[1])
    else:
        res["a"][i] = math.sqrt(4 * e[1])
        res["b"][i] = math.sqrt(4 * e[0])

class ThinSS:
    def __init__(self, img, edges_w, edges_line):
        self.temp_area_marks = None
        self.img = img
        self.shape = img.shape
        self.area_bg = None
        self.area_marks = None
        self.edges_w = edges_w
        self.edges_line = edges_line

    def marker_unbound_spread(self, edge=None):
        if edge is None:
            edge = self.area_bg*0+255
        self.area_marks[self.area_marks == -1] = 0
        self.area_marks[self.area_marks == 1] = 0
        self.area_marks[edge == 0] = 1
        self.area_marks = cv2.watershed(self.img, self.area_marks)

    def get_edge(self, edge_poly):
        edge_mask = 255 - np.zeros(self.img.shape[0:2], np.uint8)
        edge_zero = np.array([[0, 0], [self.img.shape[1], 0], [self.img.shape[1], self.img.shape[0]], [0, self.img.shape[0]]])
        cv2.fillPoly(edge_mask, pts=[edge_zero, edge_poly[0]], color=0)
        return edge_mask
        dS: int
        S: int = sum(area_marks[area_marks == 1])
        for i in range(10):
            area_marks = self.watershed_iter(area_marks, area_bg)
            dS = S - sum(area_marks[area_marks == 1])
            S = S - dS

        self.area_dist = area_dist
        self.area_sure[:] = area_sure
        self.area_unknown[:] = area_unknown
        return area_marks

    def closes2segment(self, area_bg=None, edge=None):
        if area_bg is None:
            area_bg = self.area_bg
        if edge is not None:
            self.area_bg[edge == 0] = 0
        _, area_marks = cv2.connectedComponents(self.area_bg)
        self.area_marks = area_marks + 1

    def get_bg_rsfbald(self):
        result = np.uint8((self.edges_w / self.edges_w.max()) * 255)
        ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.area_bg = 255 - result

    def add_bg_lineaments(self):
        self.area_bg = 255 - cv2.add(255 - self.area_bg, self.edges_line)

    def RunSegmentation(self, edge):
        self.get_bg_rsfbald()
        self.add_bg_lineaments()
        self.closes2segment(edge=edge)
        self.temp_area_marks = self.area_marks.copy()
        self.marker_unbound_spread(edge)

    def get_marks_areas(self):
        print("np.unique(self.area_marks, return_counts=True)")
        unique, counts = np.unique(self.area_marks, return_counts=True)
        return counts[2:-1]

    def get_marks_perimetr(self):
        masks = self.get_masks()
        for mask in masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def get_momentum(self):
        t1 = t()
        print("Sort start")
        i_s = np.argsort(self.area_marks, axis=None)
        unique, counts = np.unique(self.area_marks, return_counts=True)
        counts = np.insert(counts,0,0)
        ccnts = counts.cumsum()
        row, col = np.indices(self.area_marks.shape)
        marks = self.area_marks.flatten()[i_s]
        row = row.flatten()[i_s]
        col = col.flatten()[i_s]
        t2 = t()
        print("StartCalc, time: ", t2 - t1)

        a = np.empty((len(unique),), dtype=np.float32)
        shm = lambda: shared_memory.SharedMemory(create=True, size=a.nbytes)
        l_shm = [shm() for i in range(0, 9)]
        del a

        res = {"xC": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[0].buf),
              "yC": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[1].buf),
              "Jxx": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[2].buf),
              "Jyy": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[3].buf),
              "Jxy": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[4].buf),
              "S": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[5].buf),
              "a": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[6].buf),
              "b": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[7].buf),
              "phi": np.ndarray((len(unique),), dtype=np.float32, buffer=l_shm[8].buf)}

        def func(col, row, ccnts, res, range):
            for i in range:
                GetElementSPJ(col, row, ccnts, i, res)

        a0 = 0
        a1 = math.floor(float(len(unique))/3)
        a2 = math.floor(float(len(unique))/3*2)
        a3 = len(unique)

        tlist1 = multiprocessing.Process(target=func, args=(col, row, ccnts, res, range(a0, a1)))
        tlist2 = multiprocessing.Process(target=func, args=(col, row, ccnts, res, range(a1, a2)))
        tlist3 = multiprocessing.Process(target=func, args=(col, row, ccnts, res, range(a2, a3)))

        tlist1.start()
        tlist2.start()
        tlist3.start()

        tlist1.join()
        tlist2.join()
        tlist3.join()

        for key in res.keys():
            res[key] = res[key].copy()

        for shm in l_shm:
            shm.close()
            shm.unlink()

        t3 = t()
        print("Calc end, time: ", t3 - t2)
        return res

    def get_SP(self):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')
        M_fl = self.area_marks.flatten()
        M_fl_s = M_fl.argsort()
        U, FU, CU = np.unique(M_fl[M_fl_s], return_index=True, return_counts=True)
        Y, X = np.indices(self.area_marks.shape)
        X = X.flatten()[M_fl_s]
        Y = Y.flatten()[M_fl_s]
        S = np.empty(len(U) - 3, np.uint32)
        P = np.empty(len(U) - 3, np.uint32)
        for i in range(3, len(U)):
            X_array = X[FU[i]:FU[i] + CU[i]]
            Y_array = Y[FU[i]:FU[i] + CU[i]]
            Xmin = np.amin(X_array)
            Xmax = np.amax(X_array) + 1
            Ymin = np.amin(Y_array)
            Ymax = np.amax(Y_array) + 1
            mask = np.zeros((Ymax - Ymin + 2, Xmax - Xmin + 2), 'uint8')
            mask[Y_array-Ymin+1, X_array-Xmin+1] = 1
            S0 = np.sum(mask)
            mask = cv2.dilate(mask, kernel, iterations=1)
            P0 = np.sum(mask) - S0
            S[i - 3] = S0
            P[i - 3] = P0
        return S, P

    def area_threshold(self, th: int):
        S = self.get_marks_areas()
        B = [0 if x <= th else 1 for x in S]
        for i in range(len(S)):
            if B[i] == 0:
                self.area_marks[self.area_marks == i + 1] = 0

    def get_masks(self, area_marks=None):
        if area_marks is None:
            area_marks = self.area_marks
        masks = []
        for i in range(3, area_marks.max()):
            mask = {'segmentation': area_marks == i, 'bbox': (0, 0, 0, 0)}
            segmentation = np.where(mask['segmentation'])
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
                mask['bbox'] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                masks.append(mask)
        return masks

