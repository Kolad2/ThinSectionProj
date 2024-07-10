import math
import numpy as np
from scipy.special import erf

def pareto(x, a, xmin):
    return ((a-1)/xmin)*((x/xmin)**(-a))


def A(a):
    return ((a - 1) * np.exp(a) + 1) / (a * (a - 1))

def pdf_paretomod(X,a,scale):
    C = 1/(scale * A(a))
    Y = X.copy()
    mask1 = X >= scale
    mask2 = X < scale
    Y[mask1] = (X[mask1] / scale) ** (-a)
    Y[mask2] = (np.exp((-a) * (X[mask2] / scale - 1)))
    return C*Y

def cdf_paretomod(X,a,scale):
    mA = 1/A(a)
    X = np.array(X)
    Y = np.empty(X.shape)
    mask1 = X >= scale
    mask2 = X < scale
    Y[mask1] = 1 - (mA/(a-1)) * (X[mask1] / scale) ** (1 - a)
    Y[mask2] = mA/a*np.exp(a)*(1 - np.exp(-a * (X[mask2] / scale)))
    return Y

class paretomod:
    def __init__(self, alpha, scale):
        self.alpha = alpha
        self.scale = scale

    def pdf(self, x):
        return pdf_paretomod(x, self.alpha, self.scale)

    def cdf(self, x):
        return cdf_paretomod(x, self.alpha, self.scale)


def pdf_weibull(x, alpha, scale):
    return (alpha/scale)*((x/scale)**(alpha-1))*np.exp(-(x/scale)**alpha)

def cdf_weibull(x, alpha, scale):
    return 1 - np.exp(-(x/scale)**alpha)

class weibull:
    def __init__(self, alpha, scale):
        self.alpha = alpha
        self.scale = scale

    def pdf(self, x):
        return pdf_weibull(x, self.alpha, self.scale)

    def cdf(self, x):
        return cdf_weibull(x, self.alpha, self.scale)


def pdf_lognorm(x, sigma, scale):
    return (1/(x*sigma*np.sqrt(2*math.pi)))*np.exp(-0.5*((np.log(x/scale))/sigma)**2)

def cdf_lognorm(x, sigma, scale):
    return 0.5*(1 + erf(np.log(x/scale)/(sigma*math.sqrt(2))))

class lognorm:
    def __init__(self, sigma, scale):
        self.sigma = sigma
        self.scale = scale
        
    def pdf(self, x):
        return pdf_lognorm(x, self.sigma, self.scale)

    def cdf(self, x):
        return cdf_lognorm(x, self.sigma, self.scale)
