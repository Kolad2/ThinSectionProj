import numpy as np


def pk(x1: int, y1: int, delta_x: int, delta_y: int):
    x_ar = []
    y_ar = []
    pk = 2 * delta_y - delta_x
    x_ar.append(x1)
    y_ar.append(y1)
    for i in range(0, delta_x):
        if pk > 0:
            pk += 2 * delta_y - 2 * delta_x
            x1 += 1
            y1 += 1
        else:
            pk += 2 * delta_y
            x1 += 1
        x_ar.append(x1)
        y_ar.append(y1)
    return x_ar, y_ar


def line(x1: int, y1: int, x2: int, y2: int):
    delta_x = x2 - x1
    delta_y = y2 - y1
    print(x1)
    if delta_x > delta_y:
        x_ar, y_ar = pk(x1, y1, delta_x, delta_y)
    else:
        y_ar, x_ar = pk(y1, x1, delta_y, delta_x)
    vec = np.array([x_ar, y_ar])
    vec = np.transpose(vec)
    return vec


