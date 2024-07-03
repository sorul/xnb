from sklearn.neighbors import KernelDensity
from typing import List
from numpy import ndarray


class KDE():
  def __init__(
          self,
          feature: str,
          target_value: str,
          kernel_density: KernelDensity,
          x_points: ndarray,
          y_points: ndarray
  ) -> None:
    self.feature = feature
    self.target_value = target_value
    self.kernel_density = kernel_density
    self.X_points = x_points
    self.y_points = y_points
