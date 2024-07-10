import numpy as np


def pareto(x, a, xmin):
    return ((a-1)/xmin)*((x/xmin)**(-a))


def A(a):
    return ((a - 1) * np.exp(a) + 1) / (a * (a - 1))

def paretomodif(X,a,lam):
    C = 1/(lam*A(a))
    Y = X.copy()
    mask1 = X >= lam
    mask2 = X < lam
    Y[mask1] = (X[mask1]/lam)**(-a)
    Y[mask2] = (np.exp((-a)*(X[mask2]/lam - 1)))
    return C*Y

def Fparetomodif(X,a,scale):
    mA = 1/A(a)
    X = np.array(X)
    Y = np.empty(X.shape)
    mask1 = X >= scale
    mask2 = X < scale
    Y[mask1] = 1 - (mA/(a-1)) * (X[mask1] / scale) ** (1 - a)
    Y[mask2] = mA/a*np.exp(a)*(1 - np.exp(-a * (X[mask2] / scale)))
    return Y

def weibull(x, alpha, scale):
    return (alpha/scale)*((x/scale)**(alpha-1))*np.exp(-(x/scale)**alpha)

def Fweibull(x, alpha, scale):
    return 1 - np.exp(-(x/scale)**alpha)