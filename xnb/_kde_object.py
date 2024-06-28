from sklearn.neighbors import KernelDensity
from typing import List


class KDE():
  def __init__(
          self,
          feature: str,
          target: str,
          kernel_density: KernelDensity,
          x_points: List,
          y_points: List
  ) -> None:
    self.feature = feature
    self.target = target
    self.kernel_density = kernel_density
    self.X_points = x_points
    self.y_points = y_points
