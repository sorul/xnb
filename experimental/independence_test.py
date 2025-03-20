from pathlib import Path
from typing import Tuple
import pandas as pd

from experimental.independence import (
    independence_test,
    independence_test_2,
    independence_test_3,
    independence_test_4
)

@skip(reason='Experimental test')
def test_independence_test(benchmark):
  x, y = load_dataset(Path('data/Leukemia_GSE9476.csv'), class_column='type')
  r = benchmark(independence_test, x, y)
  assert r is not None

@skip(reason='Experimental test')
def test_independence_test_2(benchmark):
  x, y = load_dataset(Path('data/Leukemia_GSE9476.csv'), class_column='type')
  r = benchmark(independence_test_2, x, y)
  assert r is not None

@skip(reason='Experimental test')
def test_independence_test_3(benchmark):
  x, y = load_dataset(Path('data/Leukemia_GSE9476.csv'), class_column='type')
  r = benchmark(independence_test_3, x, y)
  # r = independence_test_3(x, y)
  assert r is not None

@skip(reason='Experimental test')
def test_independence_test_4(benchmark):
  x, y = load_dataset(Path('data/Leukemia_GSE9476.csv'), class_column='type')
  r = benchmark(independence_test_4, x, y)
  # r = independence_test_4(x, y)
  assert r is not None

@skip(reason='Experimental test')
def test_compare_independence_test():
  x, y = load_dataset(Path('data/Leukemia_GSE9476.csv'), class_column='type')
  r1 = independence_test(x, y)
  r2 = independence_test_2(x, y)
  r3 = independence_test_3(x, y)
  r4 = independence_test_4(x, y)
  assert r1 == r2
  assert r1 == r3
  assert r1 == r4


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
