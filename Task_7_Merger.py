import csv
import pandas as pd

def converter(FileName):
    FileName = FileName.replace("B", "Б-")
    g = FileName.find("a")
    if g != -1:
        return FileName[0:g]
    g = FileName.find("b")
    if g != -1:
        return FileName[0:g]
    return FileName

PathStat = "/media/kolad/HardDisk/StatisticData/"

with (open(PathStat + "StatisticResult.csv", newline='') as csvfile1,
      open(PathStat + "Petrograph.csv", newline='') as csvfile2,
      open(PathStat + "ResultTable.csv", 'w', encoding='UTF8', newline='') as f):
    writer = csv.writer(f)
    rows_stat = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    rows_petro = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))

    ColNames = rows_stat[0] + ["ПетрографТипы", "ТипыТектонитов", "%матрикса"]
    writer.writerow(ColNames)

    tb_petro = pd.DataFrame(rows_petro[1:], columns=rows_petro[0])
    tb_stat = pd.DataFrame(rows_stat[1:], columns=rows_stat[0])

    for row in tb_stat.values:
        FileName = converter(row[0])
        row2 = tb_petro[tb_petro["НомерОбразца"].isin([FileName])]
        C1 = (row2["ПетрографТипы"].values.tolist())[0]
        C2 = (row2["ТипыТектонитов"].values.tolist())[0]
        C3 = (row2["%матрикса"].values.tolist())[0]
        row2 = row.tolist() + [C1, C2, C3]
        writer.writerow(row2)



with (open("temp/StatisticResult.csv", newline='') as csvfile1,
      open(PathStat + "ResultTable.csv", newline='') as csvfile2,
      open("temp/ResultTable.csv", 'w', encoding='UTF8', newline='') as f):
    writer = csv.writer(f)
    rows_stat1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    rows_stat2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    tb_stat1 = pd.DataFrame(rows_stat1[1:], columns=rows_stat1[0])
    tb_stat2 = pd.DataFrame(rows_stat2[1:], columns=rows_stat2[0])
    tb_stat1 = tb_stat1.sort_values("Номер Образца")
    tb_stat2 = tb_stat2.sort_values("Номер Образца")

    Names1 = tb_stat1.columns.tolist()[1:-3]
    Names2 = tb_stat2.columns.tolist()[1:-3]
    Names3 = ["ПетрографТипы", "ТипыТектонитов", "%матрикса"]
    NamesU = ["Номер Образца"] + Names1 + Names2 + Names3
    writer.writerow(NamesU)

    N = tb_stat1.shape[0]
    for i in range(0,N):
        Name = tb_stat1["Номер Образца"].values[i]
        v1 = list((tb_stat1[tb_stat1["Номер Образца"] == Name])[Names1].values[0])
        v2 = list((tb_stat2[tb_stat2["Номер Образца"] == Name])[Names2].values[0])
        v3 = list((tb_stat2[tb_stat2["Номер Образца"] == Name])[Names3].values[0])
        v4 = [Name] + v1 + v2 + v3
        writer.writerow(v4)