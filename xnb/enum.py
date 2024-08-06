"""Enum classes used in Explicable Naive Bayes."""
from enum import Enum


class BWFunctionName(Enum):
  """Enum class to select the bandwidth function."""

  HSILVERMAN = 'hsilverman'
  HSCOTT = 'hscott'
  HSJ = 'hsj'
  BEST_ESTIMATOR = 'best_estimator'


class Kernel(str, Enum):
  """Enum class to select the kernel function."""

  GAUSSIAN = 'gaussian'
  COSINE = 'cosine'
  EPANECHNIKOV = 'epanechnikov'
  TOPHAT = 'tophat'
  EXPONENTIAL = 'exponential'
  LINEAR = 'linear'


class Algorithm(str, Enum):
  """Enum class to select the algorithm."""

  KD_TREE = 'kd_tree'
  BALL_TREE = 'ball_tree'
  AUTO = 'auto'
