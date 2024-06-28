'''
Functions ported from the R package sm.

Implements different bandwidth selection methods, including:
- Scott's rule of thumb
- Silverman's rule of thumb
- Sheather-Jones estimator
'''

import numpy as np
from scipy.stats import norm
import scipy.interpolate as interpolate
from pandas import Series
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

__all__ = ['wmean',
           'wvar',
           'dnorm',
           'hsilverman',
           'hscott',
           'hnorm',
           'hsj',
           'best_estimator']


def wmean(x, w):
  '''
  Weighted mean
  '''
  return sum(x * w) / float(sum(w))


def wvar(x, w):
  '''
  Weighted variance
  '''
  return sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)


def dnorm(x):
  return norm.pdf(x, 0.0, 1.0)


def bowman(x):
  pass
  # TODO: implement?
  # hx = median(abs(x - median(x))) / 0.6745 * (4 / 3 / r.n) ^ 0.2
  # hy = median(abs(y - median(y))) / 0.6745 * (4 / 3 / r.n) ^ 0.2
  # h = sqrt(hy * hx)


def _select_sigma(x):
  """
  Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.
  References
  ----------
  Silverman (1986) p.47
  """
  # normalize = norm.ppf(.75) - norm.ppf(.25)
  normalize = 1.349
  IQR = (scoreatpercentile(x, 75) - scoreatpercentile(x, 25)) / normalize
  std_dev = np.std(x, axis=0, ddof=1)
  if IQR > 0:
    return np.minimum(std_dev, IQR)
  elif std_dev > 0:
    return std_dev
  else:
    return 0.0000001


def empiricalcdf(data, method='Hazen'):
  """Return the empirical cdf.
  Methods available:
      Hazen:       (i-0.5)/N
      Weibull:     i/(N+1)
      Chegodayev:  (i-.3)/(N+.4)
      Cunnane:     (i-.4)/(N+.2)
      Gringorten:  (i-.44)/(N+.12)
      California:  (i-1)/N
  Where i goes from 1 to N.
  """

  i = np.argsort(np.argsort(data)) + 1.
  N = len(data)
  method = method.lower()
  if method == 'hazen':
    cdf = (i-0.5)/N
  elif method == 'weibull':
    cdf = i/(N+1.)
  elif method == 'california':
    cdf = (i-1.)/N
  elif method == 'chegodayev':
    cdf = (i-.3)/(N+.4)
  elif method == 'cunnane':
    cdf = (i-.4)/(N+.2)
  elif method == 'gringorten':
    cdf = (i-.44)/(N+.12)
  else:
    raise ValueError('Unknown method. Choose among Weibull, Hazen,'
                     'Chegodayev, Cunnane, Gringorten and California.')

  return cdf


def scoreatpercentile(data, percentile):
  """Return the score at the given percentile of the data.
  Example:
      >>> data = randn(100)
          >>> scoreatpercentile(data, 50)
      will return the median of sample `data`.
  """
  per = np.array(percentile)
  cdf = empiricalcdf(data)
  interpolator = interpolate.interp1d(np.sort(cdf), np.sort(data))
  return interpolator(per/100.)


def best_estimator(data: Series, x_sample) -> float:
  range = abs(max(data) - min(data))
  len_unique = len(data.unique())
  params = {'bandwidth': np.linspace(range/len_unique, range, x_sample)}
  data = data.values[:, np.newaxis]
  grid = GridSearchCV(KernelDensity(), params, cv=3)
  grid.fit(list(data))
  return float(grid.best_estimator_.bandwidth_)


def hsilverman(x, x_sample):
  A = _select_sigma(x)
  n = len(x)
  return .9 * A * n ** (-0.2)


def hscott(x, x_sample):
  A = _select_sigma(x)
  n = len(x)
  return 1.059 * A * n ** (-0.2)


def hnorm(x, weights=None) -> float:
  '''
  Bandwidth estimate assuming f is normal. See paragraph 2.4.2 of
  Bowman and Azzalini[1]_ for details.

  References
  ----------
  .. [1] Applied Smoothing Techniques for Data Analysis: the
      Kernel Approach with S-Plus Illustrations.
      Bowman, A.W. and Azzalini, A. (1997).
      Oxford University Press, Oxford
  '''

  x = np.asarray(x)

  if weights is None:
    weights = np.ones(len(x))

  n = float(sum(weights))

  if len(x.shape) == 1:
    sd = np.sqrt(wvar(x, weights))
    return sd * (4 / (3 * n)) ** (1 / 5.0)

  # TODO: make this work for more dimensions
  # ((4 / (p + 2) * n)^(1 / (p+4)) * sigma_i
  if len(x.shape) == 2:
    ndim = x.shape[1]
    sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
    return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd

  return 0.0


def hsj(x):
  '''
  Sheather-Jones bandwidth estimator [1]_.

  References
  ----------
  .. [1] A reliable data-based bandwidth selection method for kernel
      density estimation. Simon J. Sheather and Michael C. Jones.
      Journal of the Royal Statistical Society, Series B. 1991
  '''

  h0 = hnorm(x)
  v0 = sj(x, h0)

  if v0 > 0:
    hstep = 1.1
  else:
    hstep = 0.9

  h1 = h0 * hstep
  v1 = sj(x, h1)

  while v1 * v0 > 0:
    h0 = h1
    v0 = v1
    h1 = h0 * hstep
    v1 = sj(x, h1)

  return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))


def sj(x, h):
  '''
  Equation 12 of Sheather and Jones [1]_

  References
  ----------
  .. [1] A reliable data-based bandwidth selection method for kernel
      density estimation. Simon J. Sheather and Michael C. Jones.
      Journal of the Royal Statistical Society, Series B. 1991
  '''
  def phi6(x): return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * dnorm(x)
  def phi4(x): return (x ** 4 - 6 * x ** 2 + 3) * dnorm(x)

  n = len(x)
  one = np.ones((1, n))

  lam = np.percentile(x, 75) - np.percentile(x, 25)
  a = 0.92 * lam * n ** (-1 / 7.0)
  b = 0.912 * lam * n ** (-1 / 9.0)

  W = np.tile(x, (n, 1))
  W = W - W.T

  W1 = phi6(W / b)
  tdb = np.dot(np.dot(one, W1), one.T)
  tdb = -tdb / (n * (n - 1) * b ** 7)

  W1 = phi4(W / a)
  sda = np.dot(np.dot(one, W1), one.T)
  sda = sda / (n * (n - 1) * a ** 5)

  alpha2 = 1.357 * (abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)

  W1 = phi4(W / alpha2)
  sdalpha2 = np.dot(np.dot(one, W1), one.T)
  sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

  return (norm.pdf(0, 0, np.sqrt(2)) /
          (n * abs(sdalpha2[0, 0]))) ** 0.2 - h
