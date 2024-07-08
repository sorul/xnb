from sklearn.naive_bayes import GaussianNB
from pathlib import Path
import sklearn.model_selection as model_selection
from typing import List, Dict, Callable, Set, Tuple
from collections import defaultdict
from sklearn.metrics import accuracy_score
import pandas as pd
from itertools import product
from sklearn.neighbors import KernelDensity
import itertools
import pytest
import time

from xnb.explicable_naive_bayes import XNB, KDE
from xnb.enum import BandwidthFunction, Kernel, Algorithm
# from xnb_cayetano.explicable_naive_bayes import XNB
# from xnb_cayetano._kde_object import KDE


def getKDES(x: pd.DataFrame, y: pd.Series) -> list[KDE]:
  comb = list(itertools.product(x.columns, list(set(y))))
  kde_list = []
  x_points = [i for i in range(10)]
  y_points = x_points
  dummy_kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
  for v, c in comb:
    kde_list.append(KDE(
        feature=v,
        target_value=c,
        kernel_density=dummy_kde,
        x_points=x_points,
        y_points=y_points
    ))
  return kde_list


def test_accuracy_benchmark_naive_bayes():
  """
  poetry run pytest -k test_accuracy_benchmark_naive_bayes
  """
  x, y = load_dataset(Path('data/iris.csv'))
  accuracy_list = {'xnb': [], 'nb1': [], 'nb2': []}
  n_features_selected = []
  skf = model_selection.StratifiedKFold(
      n_splits=5, shuffle=True, random_state=0)
  for train_index, test_index in skf.split(x, y):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
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
  _, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()
  d = xnb._calculate_target_representation(
      target_col=pd.Series(y),
      class_values=set(y)
  )
  assert len(d) == len(set(y))
  for k in d:
    assert d[k] > 0 and d[k] < 1


def test_calculate_bw_function(benchmark):
  """
  poetry run pytest -k test_calculate_bw_function -v
  """
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BandwidthFunction.BEST_ESTIMATOR, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


"""
> poetry run pytest -k bandwidth -v
"""


def test_calculate_bandwidth_best_estimator(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BandwidthFunction.BEST_ESTIMATOR, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hscott(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BandwidthFunction.HSCOTT, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hsilverman(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BandwidthFunction.HSILVERMAN, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hjs(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BandwidthFunction.HSJ, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_kde(benchmark):
  """
  > poetry run pytest -k kde -v
  """
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()
  bw = xnb._calculate_bandwidth(
      x, y, BandwidthFunction.HSJ, 50, set(y)
  )

  kde_list = benchmark(
      xnb._calculate_kdes,
      x, y, Kernel.GAUSSIAN, Algorithm.AUTO, bw, 50, set(y)
  )

  assert len(kde_list) > 0


def test_calculate_divergence(benchmark):
  """
  > poetry run pytest -k test_calculate_divergence -v
  """
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()
  bw = xnb._calculate_bandwidth(
      x, y, BandwidthFunction.HSJ, 50, set(y)
  )
  kde_list = xnb._calculate_kdes(
      x, y, Kernel.GAUSSIAN, Algorithm.AUTO, bw, 50, set(y)
  )

  ranking = benchmark(xnb._calculate_divergence, kde_list)

  assert len(ranking) > 0


def test_calculate_feature_selection(benchmark):
  """
  > poetry run pytest -k test_calculate_feature_selection -v
  """
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()
  bw = xnb._calculate_bandwidth(
      x, y, BandwidthFunction.HSJ, 50, set(y)
  )
  kde_list = xnb._calculate_kdes(
      x, y, Kernel.GAUSSIAN, Algorithm.AUTO, bw, 50, set(y)
  )
  ranking = xnb._calculate_divergence(kde_list)

  d = benchmark(xnb._calculate_feature_selection, ranking, set(y))

  assert sorted(list(d.keys())) == sorted(list(set(y)))


def test_calculate_feature_selection_dict(benchmark):
  """
  > poetry run pytest -k test_calculate_feature_selection_dict -v
  """
  _, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()
  hellinger_dict = defaultdict(set)
  stop_dict = defaultdict(dict[str, bool])
  for cv_1, cv_2 in product(set(y), repeat=2):
    if cv_1 != cv_2:
      stop_dict[cv_1][cv_2] = False

  hellinger_dict, stop_dict = benchmark(
      xnb._calculate_feature_selection_dict,
      hellinger_dict,
      stop_dict,
      feature='petal_length',
      hellinger=0.999,
      threshold=0.05,
      class_a='virginica',
      class_b='versicolor'
  )
  assert len(hellinger_dict) > 0
  assert len(stop_dict) > 0


def load_dataset(
        file_path: Path,
        class_column: str = 'class',
        n_cols=10,
        sep=','
) -> Tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv(file_path, sep=sep).drop('samples', axis=1, errors='ignore')
  y = df[class_column]
  x = df.drop(class_column, axis=1)
  x = x[list(x.columns)[:n_cols]]
  return x, y
