import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors
import random
import matplotlib as mpl

x = np.linspace(0,10,100)
y1 = x ** 2
y2 = x ** 3
y3 = x ** (2.5)
fig = plt.figure(figsize=(6, 10))
ax = [fig.add_subplot(1, 1, 1)]

ax[0].fill_between(x, y1, y2, color='gray', label='1')
ax[0].plot(x, y1, color='black', linestyle='--', label='2')
ax[0].plot(x, y2, color='black', linestyle='-.', label='3')
ax[0].plot(x, y3, color='black', linestyle='-', label='4')
ax[0].set_xlabel("$X_{n}$")
ax[0].set_ylabel("$H_{\phi}$")
ax[0].legend()
plt.show()


exit()
def GetParam(tb,Key):
    res = pd.to_numeric(tb[Key]).values
    if len(res) != 0:
        return (res)[0]
    else:
        return None
RNAME = ["Horga","Marble","Elanci","Talovka","Tonta"]#2900 таловка
k = 0
with (open('temp/StatisticResult.csv', newline='') as f,
      open('temp/PetroBinding_' + RNAME[k] + '.csv', newline='') as f2):
    rows_stat = list(csv.reader(f, delimiter=',', quotechar='|'))
    petro_bind = list(csv.reader(f2, delimiter=',', quotechar='|'))
    tb_stat = pd.DataFrame(rows_stat[1:], columns=rows_stat[0])
    tb_bind = pd.DataFrame(petro_bind[1:], columns=petro_bind[0])

N = len(tb_bind)

Names = [None for i in range(0, N)]
X_et = np.empty(N, np.float16)
X_n = np.empty(N, np.float16)
H_phi = np.empty((N,2), np.float16)
H_r = np.empty((N,2), np.float16)
mu = np.empty((N,2), np.float16)
s = np.empty((N,2), np.float16)


# dict = scipy.io.loadmat("temp/Tonta_el.mat", squeeze_me=True)
# print(dict["grid_x"])
# fig = plt.figure(figsize=(14, 9))
# ax = [fig.add_subplot(1, 1, 1)]
# ax[0].pcolor(dict["grid_x"], dict["grid_y"], dict["grid_z"])
# fig.suptitle("Tonta2", fontsize=16)
# fig.savefig("temp/Tonta2.png")
# plt.show()
# exit()


for i in range(0, N):
    Names[i] = tb_bind["Номер_образца"].values[i]
    if(Names[i][1] == 'Б'):
        Name2 = "B21-" + Names[i][6:-1]
    else:
        Name2 = Names[i][1:-1]
    b0 = (tb_stat["Номер Образца"] == (Name2))
    b1 = (tb_stat["Номер Образца"] == (Name2 + "a"))
    b2 = (tb_stat["Номер Образца"] == (Name2 + "b"))
    X_et[i] = tb_bind["Дистанция_по_ЭТ"].values[i]
    X_n[i] = pd.to_numeric(tb_bind["Дистанция_нормаль"].values[i])

    H_phi[i, 0] = GetParam(tb_stat[b1], "H_phi")
    H_phi[i, 1] = GetParam(tb_stat[b2], "H_phi")
    if np.isnan(H_phi[i, 0]):
        H_phi[i, 0] = GetParam(tb_stat[b0], "H_phi")
        if np.isnan(H_phi[i, 0]):
            print("Данных нет")

    H_r[i, 0] = GetParam(tb_stat[b1], "H_r")
    H_r[i, 1] = GetParam(tb_stat[b2], "H_r")
    if np.isnan(H_r[i, 0]):
        H_r[i, 0] = GetParam(tb_stat[b0], "H_r")
        if np.isnan(H_r[i, 0]):
            print("Данных нет")

    mu[i, 0] = GetParam(tb_stat[b1], "smu")
    mu[i, 1] = GetParam(tb_stat[b2], "smu")
    if np.isnan(mu[i, 0]):
        mu[i, 0] = GetParam(tb_stat[b0], "smu")
        if np.isnan(mu[i, 0]):
            print("Данных нет")

    s[i, 0] = GetParam(tb_stat[b1], "s")
    s[i, 1] = GetParam(tb_stat[b2], "s")
    if np.isnan(s[i, 0]):
        s[i, 0] = GetParam(tb_stat[b0], "s")
        if np.isnan(s[i, 0]):
            print("Данных нет")


cmap = np.array([[0, 0, 128, 255],
                 [0, 0, 170, 255],
                 [0, 0,  211, 255],
                 [0, 0,  255, 255],
                 [0, 128,  255, 255],
                 [0, 255,  255, 255],
                 [0, 192, 128, 255],
                 [0, 255, 0, 255],
                 [0, 128, 0, 255],
                 [128, 192, 0, 255],
                 [255, 255, 0, 255],
                 [191, 128, 0, 255],
                 [255, 128, 0, 255],
                 [255, 0, 0, 255],
                 [211, 0, 0, 255],
                 [132, 0, 64, 255],
                 [96, 0, 96,  255]])/255
cmap = np.flipud(cmap)
cmap = mpl.colors.ListedColormap(cmap)
dict = scipy.io.loadmat("temp/" + RNAME[k] + "_el.mat", squeeze_me=True)

fig = plt.figure(figsize=(6, 10))
ax = [fig.add_subplot(6, 1, 1),
      fig.add_subplot(6, 1, 2),
      fig.add_subplot(6, 1, 3),
      fig.add_subplot(6, 1, 4),
      fig.add_subplot(6, 1, (5,6))]

ax[0].plot(X_et, H_phi,'.')
ax[0].set_xlabel("$X_{n}$")
ax[0].set_ylabel("$H_{\phi}$")

ax[1].plot(X_et, H_r,'.')
ax[1].set_xlabel("$X_{n}$")
ax[1].set_ylabel("$H_{r}$")
ax[1].sharex(ax[0])

ax[2].plot(X_et, mu, '.')
ax[2].set_xlabel("$X_{n}$")
ax[2].set_ylabel("$exp(\mu)$")
ax[2].sharex(ax[0])

ax[3].plot(X_et, s, '.')
ax[3].set_xlabel("$X_{n}$")
ax[3].set_ylabel("$s$")
ax[3].sharex(ax[0])

im = ax[4].pcolor(dict["grid_x"], dict["grid_y"], dict["grid_z"], cmap=cmap, vmin=0.5, vmax=6.0)
ax[4].sharex(ax[0])
#ax[4].set_ylim([400, 1100])
#ax[0].set_xlim([0, 2200])
ax[0].set_xlim([0, 1250])
#ax[0].set_xlim([3000, 4200])
#ax[0].set_xlim([0, 1500])
ax[0].set_title(RNAME[k])

#fig.colorbar(im, ax=ax[4], location='right', anchor=(0, 0.3))
print(np.nanmin(dict["grid_z"]))
print(np.nanmax(dict["grid_z"]))

fig.savefig("temp/" + RNAME[k] + "_gr.png")
plt.show()
exit()


H_phi = np.empty((N,2), np.float16)
H_r = np.empty((N,2), np.float16)
mu = np.empty((N,2), np.float16)
s = np.empty((N,2), np.float16)


Names = [None for i in range(0,N)]
X = np.arange(0, 5, 0.5, dtype=int)

for i, Name in enumerate(Array1):
    b1 = (tb_stat["Номер Образца"] == Name + "a")
    b2 = (tb_stat["Номер Образца"] == Name + "b")
    key = "H_r"
    H_phi[i, 0] = GetParam(b1, key)
    H_phi[i, 1] = GetParam(b2, key)
    key = "H_phi"
    H_r[i, 0] = GetParam(b1, key)
    H_r[i, 1] = GetParam(b2, key)
    key = "Lognorm mu"
    mu[i, 0] = GetParam(b1, key)
    mu[i, 1] = GetParam(b2, key)
    key = "Lognorm s"
    s[i, 0] = GetParam(b1, key)
    s[i, 1] = GetParam(b2, key)
    Names[i] = Array1[0:5]




fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(5, 1, 1),
      fig.add_subplot(5, 1, 2),
      fig.add_subplot(5, 1, 3),
      fig.add_subplot(5, 1, 4),
      fig.add_subplot(5, 1, 5)]

ax[0].plot(Names, H_phi,'.')
ax[0].set_xlabel("%матрикса")
ax[0].set_ylabel("$H_{\phi}$")

ax[1].plot(H_r,'.')
ax[1].set_xlabel("%матрикса")
ax[1].set_ylabel("$H_{r}$")

ax[2].plot(mu,'.')
ax[2].set_xlabel("%матрикса")
ax[2].set_ylabel("Lognorm $\mu$")

ax[3].plot(s,'.')
ax[3].set_xlabel("%матрикса")
ax[3].set_ylabel("Lognorm $s$")


#ax[4].set_xlabel("%матрикса")
#ax[4].set_ylabel("Lognorm $s$")

plt.show()
exit()
#tb_stat = tb_stat[tb_stat["ПетрографТипы"] == "Кварцитосланцы"]

with open('temp/ResultTable.csv', newline='') as f:
    rows_stat = list(csv.reader(f, delimiter=',', quotechar='|'))
    tb_stat = pd.DataFrame(rows_stat[1:], columns=rows_stat[0])

bools = [None, None, None]
bools[0] = ((tb_stat["ТипыТектонитов"] == "Милонит") | (tb_stat["ТипыТектонитов"] == "Ультрамилонит") | (tb_stat["ТипыТектонитов"] == "Бластомилонит"))
bools[1] = ((tb_stat["ТипыТектонитов"] == "Катаклазит") | (tb_stat["ТипыТектонитов"] == "Бластокатаклазит"))
bools[2] = (tb_stat["ТипыТектонитов"] == "Вмещающая порода")

TrueArray = (tb_stat["Lognorm boolean"] == "True")

X = pd.to_numeric(tb_stat["%матрикса"]).values
Y_rphi = pd.to_numeric(tb_stat["H_drdphi"]).values
Y_r = pd.to_numeric(tb_stat["H_dr"]).values
Y_dphi = pd.to_numeric(tb_stat["H_dphi"]).values
Y_phi = pd.to_numeric(tb_stat["H_phi"]).values
c_med = pd.to_numeric(tb_stat["c_med"]).values
mu = pd.to_numeric(tb_stat["Lognorm mu"]).values
s = pd.to_numeric(tb_stat["Lognorm s"]).values
dr_med = pd.to_numeric(tb_stat["dr_med"]).values
H_r = pd.to_numeric(tb_stat["H_r"]).values

fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(4, 1, 1),
      fig.add_subplot(4, 1, 2),
      fig.add_subplot(4, 1, 3),
      fig.add_subplot(4, 1, 4)]
f1 = mu
ax[0].plot(X, f1,'.')
ax[0].set_xlabel("%матрикса")
ax[0].set_ylabel("$H_{\Delta r,\Delta \phi}$")

f2 = Y_r
ax[1].plot(X[bools[0]], f2[bools[0]],'.', color='r')
ax[1].plot(X[bools[1]], f2[bools[1]],'.', color='b')
ax[1].plot(X[bools[2]], f2[bools[2]],'.', color='g')
ax[1].set_xlabel("%матрикса")
ax[1].set_ylabel("$H_{\Delta r}$")

f3 = c_med
ax[2].plot(X, f3, '.')
ax[2].set_xlabel("%матрикса")
ax[2].set_ylabel("$H_{\Delta\phi}$")

f4 = Y_phi*c_med
ax[3].plot(X, f4,'.')
ax[3].set_xlabel("%матрикса")
ax[3].set_ylabel("$H_{\phi}$")
plt.show()



fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 1, 1)]
X = Y_phi
Y = Y_r
ax[0].plot(X[bools[0]], Y[bools[0]],'.', color='r')
ax[0].plot(X[bools[1]], Y[bools[1]],'.', color='b')
ax[0].plot(X[bools[2]], Y[bools[2]],'.', color='g')
#ax[0].plot(X, Y_phi,'.')
#ax[0].set_xlabel("$\mu$")
#ax[0].set_ylabel("$dr_{median}$")
plt.show()

exit()

fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].plot(Y_r[TrueArray], Y_dphi[TrueArray],'.')
ax[0].set_xlabel("$H_{\Delta r}$")
ax[0].set_ylabel("$H_{\Delta \phi}$")
plt.show()

exit()
Path0 = "/media/kolad/HardDisk/StatisticData/StatMatrData/"
FileNames = os.listdir(Path0)
FilePath = Path0 + FileNames[8]

dict = scipy.io.loadmat(FilePath, squeeze_me=True)
bins = np.linspace(0,180,120)

dict["phi"][dict["phi"] < 0] = 180 + dict["phi"][dict["phi"] < 0]

array = dict["S"] > 20

dict["phi"] = np.array(dict["phi"][array])
dict["x"] = np.array(dict["x"][array])
dict["y"] = np.array(dict["y"][array])

f, _ = np.histogram(dict["phi"], bins=bins, density=True)

X = np.column_stack([dict["x"], dict["y"]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, ind = nbrs.kneighbors(X)
distances = distances[:,1]
ind = ind[:,1]

def Getdphi(phi1: np.ndarray, phi2: np.ndarray):
    dphi = np.abs(phi1-phi2)
    return np.min([dphi, 180 - dphi], axis=0)

dphi1 = Getdphi(dict["phi"], dict["phi"][ind])
dphi2 = Getdphi(dict["phi"], np.random.permutation(dict["phi"].copy()))


bins_dr = np.linspace(0,80,120)
f_dr, _ = np.histogram(distances, bins_dr, density=True)
bins_dphi = np.linspace(0, 90, 120)
f1, _ = np.histogram(dphi1, bins_dphi, density=True)
f2, _ = np.histogram(dphi2, bins_dphi, density=True)

f_drdphi,_,_ = np.histogram2d(distances, dphi1, bins=(bins_dr, bins_dphi), density=True)


def GetH(P):
    return -np.sum(P * np.log2(P, out=np.zeros_like(P), where=(P != 0)), axis=None)


H_drdphi = GetH(f_drdphi*bins_dr[1]*bins_dphi[1])
H_dr = GetH(f_dr*bins_dr[1])
H_dphi = GetH(f1*bins_dphi[1])



fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(3, 2, 1),
      fig.add_subplot(3, 2, 3),
      fig.add_subplot(3, 2, 5),
      fig.add_subplot(1, 2, 2)]
ax[0].stairs(f_dr,bins_dr)
ax[0].set_title("Распределение ближайшего соседа")
ax[1].stairs(f1, bins_dphi)
ax[1].set_title("Разница углов ближайшего соседа")
ax[2].stairs(f2, bins_dphi)
ax[2].set_title("Разница углов перемешанных")
ax[3].pcolor(bins_dphi, bins_dr, f_drdphi)
#ax[3].plot(distances,dphi1,'.')
ax[3].set_ylabel("Расстояния")
ax[3].set_xlabel("Углы")
plt.show()

exit()
fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 1, 1, projection='polar')]
#ax[0].set_yscale('log')
#ax[0].stairs(f,bins)
ax[0].bar(bins[0:-1]/180*np.pi, f)
plt.show()
print(dict["phi"])

exit()
Types = tb["ТипыТектонитов"].unique()
print(Types)
bools = [None, None, None]
bools[0] = ((tb["ТипыТектонитов"] == "Милонит") | (tb["ТипыТектонитов"] == "Ультрамилонит") | (tb["ТипыТектонитов"] == "Милонит"))
bools[1] = ((tb["ТипыТектонитов"] == "Катаклазит") | (tb["ТипыТектонитов"] == "Бластокатаклазит"))
bools[2] = (tb["ТипыТектонитов"] == "Вмещающая порода")
Distr = {}
l_tb = tb[bools[0]]
mu = pd.to_numeric(l_tb["Lognorm mu"]).values
s = pd.to_numeric(l_tb["Lognorm s"]).values
Matr = pd.to_numeric(l_tb["%матрикса"]).values
Distr["Милонит"] = {"mu": mu, "s": s}


l_tb = tb[bools[1]]
mu = pd.to_numeric(l_tb["Lognorm mu"]).values
s = pd.to_numeric(l_tb["Lognorm s"]).values
Matr = pd.to_numeric(l_tb["%матрикса"]).values
Distr["Катаклозит"] = {"mu": mu, "s": s}




l_tb = tb[bools[2]]
mu = pd.to_numeric(l_tb["Lognorm mu"]).values
s = pd.to_numeric(l_tb["Lognorm s"]).values
Matr = pd.to_numeric(l_tb["%матрикса"]).values
Distr["Вмещающая порода"] = {"mu": mu, "s": s}


fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].plot(Distr["Милонит"]["mu"], Distr["Милонит"]["s"],'*', color='black')
ax[0].plot(Distr["Катаклозит"]["mu"], Distr["Катаклозит"]["s"],'o', color='black')
ax[0].plot(Distr["Вмещающая порода"]["mu"], Distr["Вмещающая порода"]["s"],'+', color='black')
ax[0].set_xlabel("Lognorm mu")
ax[0].set_ylabel("Lognorm s")
#ax[0].set_ylabel("%матрикса")
plt.show()
exit()