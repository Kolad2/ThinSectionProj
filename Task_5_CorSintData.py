import os
import sys
import scipy.io
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import math
from scipy.ndimage import histogram
from scipy import stats as st
from scipy.special import factorial
from scipy.special import gamma
import scipy.stats as st
import scipy.optimize as opt
from scipy.optimize import minimize


FileNames = ["B21-234a",    #0
             "B21-215b",    #1
             "B21-215a",    #2
             "B21-213b",    #3
             "B21-213a",    #4
             "B21-189b",    #5
             "B21-188b_2",  #6
             "B21-151b",    #7
             "B21-107b",    #8
             "64-3",        #9
             "15-2",        #10
             "15-1",        #11
             "B21-166b",    #12
             "B21-122a",    #13
             "B21-120a",    #14
             "B21-51a",     #15
             "B21-200b",    #16
             "B21-192b",    #17
             "19-5b"]       #18

Path0 = "/media/kolad/HardDisk/StatisticData/"
FileNames = os.listdir(Path0 + "StatisticSintData")
intermax = 500

for FileName in FileNames:
    print(FileName)
    SintFile = Path0 + "StatisticSintData/" + FileName + "/" + FileName + "_S.mat"
    CorFile = Path0 + "StatisticCorData/" + FileName + "/" + FileName + "_S.mat"
    if os.path.exists(CorFile):
        print("Файл коррекции существует, пропуск")
        continue
    matdict = scipy.io.loadmat(SintFile, squeeze_me=True)
    S = matdict['S']
    P = matdict['P']
    hS = [[] for i in range(0, intermax)]
    for j in range(0, intermax):
        mask = S[j] > 5
        #print('S',len(S[j]))
        #print('P',len(P[j]))
        S[j] = S[j][mask]
        P[j] = P[j][mask]
        hS[j] = np.empty(len(S[j]), dtype=float)
        hS[j] = S[j] + P[j]*st.uniform.rvs(size=len(P[j]))
        #for i in range(0,len(S[j])):
        #    hS[j][i] = S[j][i] + np.sum(st.uniform.rvs(size=P[j][i]))
    dict = {"S": hS, "P": P}
    if not os.path.exists(Path0 + "StatisticCorData/" + FileName):
        os.mkdir(Path0 + "StatisticCorData/" + FileName)
    scipy.io.savemat(CorFile, dict)