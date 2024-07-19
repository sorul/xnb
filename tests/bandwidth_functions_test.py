from pathlib import Path
from typing import Tuple
import pandas as pd

from xnb.explicable_naive_bayes import XNB
from xnb.enum import BWFunctionName


"""
poetry run pytest -k bandwidth -v
"""


def test_calculate_bandwidth_best_estimator(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BWFunctionName.BEST_ESTIMATOR, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hscott(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BWFunctionName.HSCOTT, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hsilverman(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BWFunctionName.HSILVERMAN, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def test_calculate_bandwidth_hjs(benchmark):
  x, y = load_dataset(Path('data/iris.csv'))
  xnb = XNB()

  d = benchmark(
      xnb._calculate_bandwidth,
      x, y, BWFunctionName.HSJ, 50, set(y)
  )

  assert len(d.keys()) == len(set(y))


def load_dataset(
        file_path: Path,
        class_column: str = 'class',
        n_cols: int = 10,
        sep: str = ','
) -> Tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv(file_path, sep=sep).drop('samples', axis=1, errors='ignore')
  y = df[class_column]
  x = df.drop(class_column, axis=1)
  x = x[list(x.columns)[:n_cols]]
  return x, y
