import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
import math
from random import shuffle
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad

class ThinSegmentation:
    def __init__(self, img, edges_w=None, edges_line=None):
        self.img = img
        self.shape = img.shape
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        #l_c, _, _ = cv2.split(lab)
        #self.img_gray = l_c
        self.area_sure = None#np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_unknown = None#np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_dist = None
        self.area_bg = None#np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_marks = None
        if edges_w is not None:
            self.edges_w = edges_w
        else:
            self.edges_w = None
        if edges_line is not None:
            self.edges_line = edges_line
        else:
            self.edges_line = None


    def watershed_iter(self, area_marks, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg
        area_marks = area_marks.copy()
        area_marks[area_marks == -1] = 0
        area_marks[area_marks == 1] = 0
        area_marks[area_bg == 0] = 1
        return cv2.watershed(self.img, area_marks)

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

    def area_marks_edgeupdate(self, area_marks):
        area_marks[area_marks == -1] = 0
        area_marks[area_marks == 0] = 0
        area_marks[area_marks == 1] = 0
        area_marks[self.area_bg == 0] = 1

    def area_marks_summator(self, area_marks_base, area_marks):
        area_marks_base = area_marks_base.copy()
        self.area_marks_edgeupdate(area_marks_base)
        area_marks_base[area_marks > 1] = area_marks[area_marks > 1] + self.area_marks.max()
        return area_marks_base

    def get_marker_from_background_iter(self, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg.copy()
        else:
            area_bg = area_bg.copy()

        if self.area_marks is not None:
            area_bg[self.area_marks != 1] = 0
            area_marks = self.get_marker_from_background(area_bg)
            self.area_marks = self.area_marks_summator(self.area_marks, area_marks)
            self.area_marks = self.watershed_iter(self.area_marks)
        else:
            area_marks = self.get_marker_from_background(area_bg)
            self.area_marks = np.empty(self.shape[0:2], dtype=np.int32)
            self.area_marks = area_marks

    def get_marker_from_background(self, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg
        area_marks = np.empty(self.shape[0:2], dtype=np.int32)
        area_dist = np.empty(self.shape[0:2], dtype=np.float32)
        area_dist[:] = cv2.distanceTransform(area_bg, cv2.DIST_L2, 0)
        area_dist = np.uint8((area_dist / area_dist.max()) * 255)
        ret, area_sure = cv2.threshold(
            area_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, area_marks[:] = cv2.connectedComponents(area_sure)
        area_unknown = cv2.subtract(area_bg, area_sure)
        area_marks = area_marks + 1

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

    def closes2segment(self, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg
        _, area_marks = cv2.connectedComponents(self.area_bg)
        self.area_marks = area_marks + 1


    def get_edge_prob(self):
        if self.edges_w is None:
            self.edges_w = np.empty(self.shape[0:2], dtype=np.float32)
            model = modelini()
            self.edges_w[:] = get_model_edges(model, self.img)
        return self.edges_w

    def set_edge_prob(self, edges_w):
        self.edges_w = edges_w.copy()

    def set_edge_line(self, edges_line):
        self.edges_line = edges_line.copy()

    def get_bg_canny(self):
        self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.area_bg = 255 - cv2.Canny(self.img_gray, 100, 200)
        self.edges_w = self.get_edge_prob()
        self.area_bg[self.edges_w < 0.1] = 255
        self.get_marker_from_background_iter()
        return self.area_marks

    def get_bg_rsfcanny(self):
        #self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.edges_w = self.get_edge_prob()
        self.area_bg = 255 - cannythresh(self.edges_w)
        self.area_bg[self.edges_w < 0.1] = 255
        return self.area_marks

    def get_bg_rsfcannygrad(self):
        #self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.edges_w = self.get_edge_prob()
        self.area_bg = 255 - cannythresh_grad(self.edges_w, self.img_gray)
        self.area_bg[self.edges_w < 0.1] = 255
        return self.area_marks

    def get_bg_rsfthin(self):
        self.edges_w = self.get_edge_prob()
        result = np.uint8((self.edges_w / self.edges_w.max()) * 255)
        ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.ximgproc.thinning(result)
        result = np.uint8((result / result.max()) * 255)
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        self.area_bg = 255 - result

    def get_bg_rsfbald(self):
        self.edges_w = self.get_edge_prob()
        result = np.uint8((self.edges_w / self.edges_w.max()) * 255)
        ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.area_bg = 255 - result

    def get_bg_rsf_skeleton_base(self):
        self.edges_w = self.get_edge_prob()
        edges_w = np.uint8((self.edges_w / self.edges_w.max()) * 255)
        edges_0 = np.zeros(edges_w.shape, np.uint8)
        for i in range(10, 255, 1):
            print(i)
            kernel = np.ones((3, 3), np.uint8)
            ret, result_bin = cv2.threshold(edges_w, i, 255, cv2.THRESH_BINARY)
            result_bin = cv2.erode(result_bin, kernel, iterations=1)
            result_erod = cv2.erode(result_bin, kernel, iterations=1)
            result_erod = cv2.subtract(result_bin, result_erod)
            result_bin = cv2.add(result_bin, edges_0)
            edges = cv2.ximgproc.thinning(result_bin)
            edges = cv2.subtract(edges, 255 - result_erod)
            edges_0 = cv2.add(edges_0, edges)
        self.area_bg = 255 - edges_0

    def add_bg_lineaments(self):
        self.area_bg = 255 - cv2.add(255 - self.area_bg, self.edges_line)

    def dilate_border(self, it=5):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        self.area_bg = cv2.erode(self.area_bg, kernel, iterations=it)

    def method0(self):
        print("self.get_bg_rsfcanny()")
        self.get_bg_rsfcanny()
        print("self.get_marker_from_background_iter()")
        self.get_marker_from_background_iter()
        print("self.get_marker_from_background_iter()")
        self.get_marker_from_background_iter()
        print("marker_unbound_spread()")
        self.marker_unbound_spread()

    def method0_1(self):
        print("self.get_bg_rsf_skeleton_base()")
        self.get_bg_rsf_skeleton_base()
        print("self.get_marker_from_background_iter()")
        self.get_marker_from_background_iter()
        print("self.get_marker_from_background_iter()")
        self.get_marker_from_background_iter()
        print("marker_unbound_spread()")
        #self.marker_unbound_spread()

    def method1(self):
        self.get_bg_rsfbald()
        self.add_bg_lineaments()
        self.closes2segment()

    def method2(self, edge):
        self.method1()
        self.marker_unbound_spread(edge)


    def method3(self, it=5):
        self.get_bg_rsf_skeleton_base()
        self.dilate_border(it)
        self.closes2segment()
        self.marker_unbound_spread()


    def get_marks_areas(self):
        print("np.unique(self.area_marks, return_counts=True)")
        unique, counts = np.unique(self.area_marks, return_counts=True)
        return counts[2:-1]

    def get_marks_perimetr(self):
        masks = self.get_masks()
        for mask in masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def get_momentum(self):
        print("sort")
        i_s = np.argsort(self.area_marks,axis=None)
        unique, counts = np.unique(self.area_marks, return_counts=True)
        counts = np.insert(counts,0,0)
        ccnts = counts.cumsum()
        row, col = np.indices(self.area_marks.shape)
        marks = self.area_marks.flatten()[i_s]
        row = row.flatten()[i_s]
        col = col.flatten()[i_s]
        Jeig_result = []
        for i in range(0,len(counts)-1):
            xC = sum(col[ccnts[i]:ccnts[i+1]]) / counts[i+1]
            yC = sum(row[ccnts[i]:ccnts[i+1]]) / counts[i+1]
            Jxx = sum(1/8 + (col[ccnts[i]:ccnts[i+1]] - xC) ** 2)
            Jyy = sum(1/8 + (row[ccnts[i]:ccnts[i+1]] - yC) ** 2)
            Jxy = sum((row[ccnts[i]:ccnts[i+1]] - yC) * (col[ccnts[i]:ccnts[i+1]] - xC)) / counts[i+1]
            #Jeig = np.linalg.eig(np.array([[Jxx, Jxy], [Jxy, Jyy]]))
            #Jeig_result.append(Jeig)
        return Jxx, Jyy, Jxy, xC, yC

    def get_SP(self):
        S = []
        P = []
        print("start")
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')
        M_fl = self.area_marks.flatten()
        M_fl_s = M_fl.argsort()
        U, FU, CU = np.unique(M_fl[M_fl_s], return_index=True, return_counts=True)
        Y, X = np.mgrid[0:self.area_marks.shape[0], 0:self.area_marks.shape[1]]
        X = X.flatten()[M_fl_s]
        Y = Y.flatten()[M_fl_s]
        for i in range(0, len(U)):
            Xmin = np.amin(X[FU[i]:FU[i] + CU[i]])
            Xmax = np.amax(X[FU[i]:FU[i] + CU[i]]) + 1
            Ymin = np.amin(Y[FU[i]:FU[i] + CU[i]])
            Ymax = np.amax(Y[FU[i]:FU[i] + CU[i]]) + 1
            mask = {'segmentation': np.uint8(self.area_marks[Ymin:Ymax,Xmin:Xmax] == U[i]),
                    'bbox': (Xmin, Ymin, Xmax-Xmin, Ymax-Ymin)}
            pic = np.zeros((Ymax-Ymin + 2, Xmax-Xmin + 2), 'uint8')
            S0 = np.sum(mask['segmentation'])
            pic[1:-1, 1:-1] = mask['segmentation']
            pic = cv2.dilate(pic, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            P0 = 0
            for contour in contours:
                c = contour.reshape((contour.shape[0], 2))
                P0 = P0 + len(c)
            S.append(S0)
            P.append(P0)
        return S, P


    def area_threshold(self, th: int):
        S = self.get_marks_areas()
        B = [0 if x <= th else 1 for x in S]
        for i in range(len(S)):
            if B[i] == 0:
                self.area_marks[self.area_marks == i + 1] = 0


    def area_marks_shuffle(self):
        l = np.max(self.area_marks)
        numlist = [x for x in range(2,l+1)]
        shuffle(numlist)
        for i in range(l-1):
            self.area_marks[self.area_marks == i + 2] = numlist[i]

    def get_masks(self, area_marks=None):
        if area_marks is None:
            area_marks = self.area_marks
        masks = []
        for i in range(area_marks.max()):
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

