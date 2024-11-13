from sklearn.neighbors import KernelDensity
from numpy import ndarray


class KDE():
  """Class to represent a Kernel Density Estimation."""

  def __init__(
          self,
          feature: str,
          target_value: str,
          kernel_density: KernelDensity,
          x_points: ndarray,
          y_points: ndarray
  ) -> None:
    """Initialize the KDE object."""
    self.feature = feature
    self.target_value = target_value
    self.kernel_density = kernel_density
    self.x_points = x_points
    self.y_points = y_points
