"""Enum classes used in Explicable Naive Bayes."""
from enum import Enum


class BWFunctionName(str, Enum):
  """Enum class to select the bandwidth function."""

  HSILVERMAN = 'hsilverman'
  HSCOTT = 'hscott'
  HSJ = 'hsj'
  BEST_ESTIMATOR = 'best_estimator'

  def __str__(self) -> str:
    """Return the enum value."""
    return self.value


class Kernel(str, Enum):
  """Enum class to select the kernel function."""

  GAUSSIAN = 'gaussian'
  COSINE = 'cosine'
  EPANECHNIKOV = 'epanechnikov'
  TOPHAT = 'tophat'
  EXPONENTIAL = 'exponential'
  LINEAR = 'linear'

  def __str__(self) -> str:
    """Return the enum value."""
    return self.value


class Algorithm(str, Enum):
  """Enum class to select the algorithm."""

  KD_TREE = 'kd_tree'
  BALL_TREE = 'ball_tree'
  AUTO = 'auto'

  def __str__(self) -> str:
    """Return the enum value."""
    return self.value
