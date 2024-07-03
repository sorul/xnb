from enum import Enum

from xnb._bandwidth_functions import (
    hsilverman, hscott, hsj, best_estimator
)


class BandwidthFunction(Enum):
  HSILVERMAN = hsilverman
  HSCOTT = hscott
  HSJ = hsj
  BEST_ESTIMATOR = best_estimator


class Kernel(str, Enum):
  GAUSSIAN = 'gaussian'
  COSINE = 'cosine'
  EPANECHNIKOV = 'epanechnikov'
  TOPHAT = 'tophat'
  EXPONENTIAL = 'exponential'
  LINEAR = 'linear'


class Algorithm(str, Enum):
  KD_TREE = 'kd_tree'
  BALL_TREE = 'ball_tree'
  AUTO = 'auto'
