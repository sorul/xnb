from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KernelDensity
import itertools
import pytest
import time

from xnb_cayetano.explicable_naive_bayes import XNB
from xnb_cayetano._kde_object import KDE


def iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv("data/iris.csv", sep=',')
  X, y = df.iloc[:, 0:-1], df.iloc[:, -1]
  return X, y


def leukemia_dataset(n_cols=500) -> tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv("data/Leukemia_GSE9476.csv",
                   sep=',').drop('samples', axis=1, errors='ignore')
  X, y = df.iloc[:, 1:n_cols], df.iloc[:, 0]
  return X, y


def getBandwidths(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
  variables = X.columns
  bw = {'variable': [], 'target': [], 'bandwidth': []}
  for v in variables:
    for c in set(y):
      bw['variable'].append(v)
      bw['target'].append(c)
      bw['bandwidth'].append(0.1)
  return pd.DataFrame(bw)


def getKDES(X: pd.DataFrame, y: pd.Series) -> list[KDE]:
  comb = list(itertools.product(X.columns, list(set(y))))
  kde_list = []
  x_points = [i for i in range(10)]
  y_points = x_points
  dummy_kde = KernelDensity(kernel="gaussian", bandwidth=0.1)
  for v, c in comb:
    kde_list.append(KDE(v, c, dummy_kde, x_points, y_points))
  return kde_list


def test_accuracy_benchmark_naive_bayes():
  """
  poetry run pytest -k test_accuracy_benchmark_naive_bayes
  """
  X, y = iris_dataset()
  accuracy_list = {'xnb': [], 'nb1': [], 'nb2': []}
  n_features_selected = []
  skf = model_selection.StratifiedKFold(
      n_splits=5, shuffle=True, random_state=0)
  for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # EXPLICABLE NB
    xnb = XNB()
    xnb.fit(X_train, y_train)
    feature_selection = xnb.feature_selection_dict
    y_pred = xnb.predict(X_test)
    accuracy_list['xnb'].append(accuracy_score(y_test, y_pred))

    # ORIGINAL NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy_list['nb1'].append(accuracy_score(y_test, y_pred))

    # FEATURE SELECTION NB
    new_cols = list({x for v in feature_selection.values() for x in v})
    nb = GaussianNB()
    nb.fit(X_train[new_cols], y_train)
    y_pred = nb.predict(X_test[new_cols])
    accuracy_list['nb2'].append(accuracy_score(y_test, y_pred))
    n_features_selected.append(len(new_cols))

  results = pd.DataFrame(
      {
          'Accuracy XNB': accuracy_list['xnb'],
          'Accuracy NB1': accuracy_list['nb1'],
          'Accuracy NB2': accuracy_list['nb2'],
          '#VAR': n_features_selected
      }
  )
  xnb_mean = results['Accuracy XNB'].mean()
  nb1_mean = results['Accuracy NB1'].mean()
  nb2_mean = results['Accuracy NB2'].mean()

  assert xnb_mean >= nb1_mean - nb1_mean * 0.05
  assert xnb_mean >= nb2_mean - nb2_mean * 0.05


def test_calculate_target_representation():
  """
  poetry run pytest -k test_calculate_target_representation
  """
  X, y = iris_dataset()
  xnb = XNB()
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)
  xnb._calculate_target_representation(y)
  d = xnb._class_representation
  assert len(d) == len(set(y))
  for k in d:
    assert d[k] > 0 and d[k] < 1


@pytest.mark.benchmark(
    min_time=0.1,
    max_time=0.5,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=False
)
def test_calculate_bw_function(benchmark):
  """> poetry run pytest -k test_calculate_bw_function -v"""
  X, y = iris_dataset()
  xnb = XNB()
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)

  benchmark(xnb._calculate_bandwidth)

  assert len(xnb._bw.variable.unique()) == len(
      X.columns) and len(xnb._bw.target.unique()) == len(set(y))


"""
> poetry run pytest -k bandwidth -v
"""


def test_calculate_bandwidth_best_estimator(benchmark):
  X, y = iris_dataset()
  xnb = XNB(bw_function=XNB.BW_BEST_ESTIMATOR)
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)

  benchmark(xnb._calculate_bandwidth)

  assert len(xnb._bw.variable.unique()) == len(X.columns)
  assert len(xnb._bw.target.unique()) == len(set(y))


def test_calculate_bandwidth_hscott(benchmark):
  X, y = iris_dataset()
  xnb = XNB(bw_function=XNB.BW_HSCOTT)
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)

  benchmark(xnb._calculate_bandwidth)

  assert len(xnb._bw.variable.unique()) == len(
      X.columns) and len(xnb._bw.target.unique()) == len(set(y))


def test_calculate_bandwidth_hsilverman(benchmark):
  X, y = iris_dataset()
  xnb = XNB(bw_function=XNB.BW_HSILVERMAN)
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)

  benchmark(xnb._calculate_bandwidth)

  assert len(xnb._bw.variable.unique()) == len(
      X.columns) and len(xnb._bw.target.unique()) == len(set(y))


def test_calculate_bandwidth_hjs(benchmark):
  X, y = iris_dataset()
  xnb = XNB(bw_function=XNB.BW_HSJ)
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)

  benchmark(xnb._calculate_bandwidth)

  assert len(xnb._bw.variable.unique()) == len(
      X.columns) and len(xnb._bw.target.unique()) == len(set(y))


def test_calculate_kde(benchmark):
  """
  > poetry run pytest -k kde -v
  """
  X, y = leukemia_dataset(10)
  xnb = XNB()
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)
  xnb._bw = getBandwidths(X, y)

  benchmark(xnb._calculate_kde)

  assert len(xnb._kde_list) > 0


def test_calculate_divergence(benchmark):
  """
  > poetry run pytest -k test_calculate_divergence -v
  """
  X, y = leukemia_dataset(50)
  xnb = XNB(bw_function=XNB.BW_HSJ)
  xnb._X = X
  xnb._y = y
  xnb._class_values = set(y)
  xnb._bw = getBandwidths(X, y)
  xnb._kde_list = getKDES(X, y)

  benchmark(xnb._calculate_divergence)

  assert len(xnb._ranking_divergence) > 0
