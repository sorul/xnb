import numpy as np


def normalize(data: np.ndarray) -> np.ndarray:
  s = np.sum(data)
  if s > 0:
    data = data / s
  s = np.sum(data)
  while s > 1.0:
    data = data / s
    s = np.sum(data)
  return data


def normalize_2(data: list) -> list:
  s = sum(data)
  if s > 0:
    data = [data[i] / s for i in range(len(data))]
  else:
    data = [1.0 / len(data) for i in range(len(data))]
  s = sum(data)
  while s > 1.0:
    data = [data[i] / s for i in range(len(data))]
    s = sum(data)
  return data


def test_compare_normalize():
  data = [
      [1, 1, 1, 1],
      [1, 2, 3, 4],
  ]

  for d in data:
    r1 = normalize(d)
    r2 = normalize_2(d)
    assert sorted(r1) == sorted(r2)
