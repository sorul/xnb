from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import numpy as np
import itertools
import numba as nb


def pearson(x1, x2, x3_dummies):
  # Compute the residuals of x1 and x2 after regressing out the effect of x3
  x1_res = x1 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x1, rcond=None)[0])
  x2_res = x2 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x2, rcond=None)[0])

  # Compute the partial correlation coefficient between x1 and x2 controlling for x3
  partial_corr, p_value = pearsonr(x1_res, x2_res)
  return partial_corr, p_value


def independence_test(X, y):
  x3 = y
  count = 0
  cont_total = 0

  x3_dummies = np.column_stack([x3 == category for category in np.unique(x3)])

  dependent = [0] * len(X.columns)

  for i in range(0, len(X.columns)-1):
    for j in range(i+1, len(X.columns)):
      if not (dependent[i] == 1 and dependent[j] == 1):
        x1 = X.iloc[:, i]
        x2 = X.iloc[:, j]

        partial_corr, p_value = pearson(x1, x2, x3_dummies)
        cont_total = cont_total + 1
        if p_value < 1e-6 and abs(partial_corr) > 0.7:
          count = count + 1
          dependent[i] = 1
          dependent[j] = 1
  return sum(dependent)


def independence_test_2(X, y):
  n_features = X.shape[1]
  dependent = np.zeros(n_features, dtype=bool)
  # count = 0

  def process_feature(i):
    if dependent[i]:
      return 0
    x1 = X.iloc[:, i].values
    local_dependent = False

    for j in range(i + 1, n_features):
      if dependent[j]:
        continue

      x2 = X.iloc[:, j].values
      partial_corr, p_value = pearsonr(x1, x2)

      if p_value < 1e-6 and abs(partial_corr) > 0.7:
        dependent[j] = True
        local_dependent = True
        return 1

    if local_dependent:
      dependent[i] = True
      return 1
    return 0

  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_feature, i)
               for i in range(n_features - 1)]
    # count = sum(future.result() for future in as_completed(futures))

  return np.sum(dependent)


def pearson_3(x1, x2, x3_dummies):
  x1_res = x1 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x1, rcond=None)[0])
  x2_res = x2 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x2, rcond=None)[0])
  partial_corr, p_value = pearsonr(x1_res, x2_res)
  return partial_corr, p_value


def independence_test_3(X, y):
  X = X.values
  y = y.values
  x3 = y
  x3_dummies = np.column_stack([x3 == category for category in np.unique(x3)])
  dependent = [0] * X.shape[1]
  n_columns = X.shape[1]

  def process(i, j):
    if not (dependent[i] and dependent[j]):
      x1 = X[:, i]
      x2 = X[:, j]
      partial_corr, p_value = pearson_3(x1, x2, x3_dummies)
      if p_value < 1e-6 and abs(partial_corr) > 0.7:
        dependent[i] = 1
        dependent[j] = 1

  Parallel(n_jobs=3)(delayed(process)(i, j)
                     for i in range(n_columns) for j in range(i+1, n_columns))
  return sum(dependent)


@nb.njit
def calculate_residuals(x, x3_dummies):
  x3_dummies = x3_dummies.astype(np.float64)
  # Calcular la pseudoinversa de Moore-Penrose
  x3_pinv = np.linalg.pinv(x3_dummies)
  coef = np.dot(x3_pinv, x)
  residuals = x - np.dot(x3_dummies, coef)
  return residuals


@nb.njit
def pearson_corr(x, y):
  n = len(x)
  mx = np.mean(x)
  my = np.mean(y)
  xm = x - mx
  ym = y - my
  r_num = np.sum(xm * ym)
  r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
  r = r_num / r_den
  return r


@nb.njit
def erf(x):
  # Aproximación de la función de error (erf)
  # Coeficientes de la serie de Taylor para la aproximación de erf
  a1 = 0.254829592
  a2 = -0.284496736
  a3 = 1.421413741
  a4 = -1.453152027
  a5 = 1.061405429
  p = 0.3275911

  # Signo de x
  sign = 1 if x >= 0 else -1
  x = np.abs(x)

  # Aproximación de la función de error
  t = 1.0 / (1.0 + p * x)
  y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2)
             * t + a1) * t * np.exp(-x * x)

  return sign * y


@nb.njit
def norm_cdf(x):
  # Aproximación de la CDF de la distribución normal usando erf
  return 0.5 * (1 + erf(x / np.sqrt(2)))


@nb.njit
def p_value_from_r(r, n):
  t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
  p = 2 * (1 - norm_cdf(np.abs(t_stat)))
  return p


@nb.njit(parallel=True)
def calculate_partial_corr_matrix(X, x3_dummies):
  n, m = X.shape
  results = np.empty((m, m, 2), dtype=np.float64)

  for i in nb.prange(m):
    for j in range(i + 1, m):
      x1 = X[:, i]
      x2 = X[:, j]
      x1_res = calculate_residuals(x1, x3_dummies)
      x2_res = calculate_residuals(x2, x3_dummies)
      partial_corr = pearson_corr(x1_res, x2_res)
      p_value = p_value_from_r(partial_corr, len(x1_res))
      results[i, j, 0] = partial_corr
      results[i, j, 1] = p_value
  return results


def independence_test_4(X, y):
  x3_dummies = np.column_stack(
      [y == category for category in np.unique(y)]).astype(np.float64)
  dependent = np.zeros(len(X.columns), dtype=np.int32)

  X_np = X.values.astype(np.float64)
  results = calculate_partial_corr_matrix(X_np, x3_dummies)

  for i, j in itertools.combinations(range(len(X.columns)), 2):
    partial_corr, p_value = results[i, j]
    if p_value < 1e-6 and abs(partial_corr) > 0.7:
      dependent[i] = 1
      dependent[j] = 1

  return np.sum(dependent)
