"""Explicable Naive Bayes."""
from sklearn.neighbors import KernelDensity
from typing import Tuple, List, Dict, Set
from pandas import DataFrame, Series
from collections import defaultdict
from itertools import product
from math import sqrt
import numpy as np

from xnb._kde_object import KDE
from xnb.enum import Algorithm, BWFunctionName, Kernel
from xnb._bandwidth_functions import get_bandwidth_function
from xnb._progress_bar import progress_bar


__all__ = [
    'XNB',
    'NotFittedError'
]


class _ClassFeatureDistance:
  def __init__(self, class_value: str, feature: str, distance: float = 0.0):
    self.class_value = class_value
    self.feature = feature
    self.distance = distance

  def __hash__(self):
    return hash(self.class_value + self.feature)


class XNB:
  """Class to perform Explainable Naive Bayes."""

  def __init__(self, show_progress_bar: bool = False) -> None:
    """Initialize the Explainable Naive Bayes object.

    ## Args:
    :param show_progress_bar: Whether to show progress bars.
    """
    self.show_progress_bar = show_progress_bar

  @property
  def feature_selection_dict(self) -> Dict[str, Set[str]]:
    """Obtain the feature selection dictionary.

    Each target class (key) is associated with a set of features (values).
    """
    if hasattr(self, '_feature_selection_dict'):
      return self._feature_selection_dict
    else:
      raise NotFittedError()

  def fit(
      self,
      x: DataFrame,
      y: Series,
      bw_function: BWFunctionName = BWFunctionName.HSILVERMAN,
      kernel: Kernel = Kernel.GAUSSIAN,
      algorithm: Algorithm = Algorithm.AUTO,
      n_sample: int = 50
  ) -> None:
    """Calculate the best feature selection to be able to predict later.

    ## Args:
    :param x: DataFrame containing the input features
    :param y: Series containing the target variable
    :param bw_function: Bandwidth function to use, defaults to
    BWFunctionName.HSILVERMAN
    :param kernel: Kernel function to use, defaults to Kernel.GAUSSIAN
    :param algorithm: Algorithm to use for KDE, defaults to Algorithm.AUTO
    :param n_sample: Number of samples to use, defaults to 50

    ## Returns:
    :return: None
    """
    class_values = set(y)
    bw_dict = self._calculate_bandwidth(
        x, y, bw_function, n_sample, class_values)
    kde_list = self._calculate_kdes(
        x, y, kernel, algorithm, bw_dict, n_sample, class_values)
    ranking = self._calculate_divergence(kde_list)
    self._calculate_feature_selection(ranking, class_values)
    self._calculate_necessary_kde(x, y, bw_dict, kernel, algorithm)
    self._calculate_target_representation(y, class_values)

  def predict(self, x: DataFrame) -> np.ndarray:
    """Return the predicted class for each row in the DataFrame.

    ## Args:
    :param x: DataFrame containing the input to predict

    ## Returns:
    :return: Numpy array containing the predicted class for each row
    """
    cond1 = not hasattr(self, '_kernel_density_dict')
    cond2 = not hasattr(self, '_class_representation')

    if cond1 or cond2:
      raise NotFittedError()

    y_pred = []
    p_len = len(x)
    p_title = 'Calculating the prediction'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for _, row in x.iterrows():
        # Calculating the probability of each class
        y, m, s = None, -np.inf, 0
        # Running through the final features
        for class_value, features in self.feature_selection_dict.items():
          pr = 0
          for feature in features:
            # We get the probabilities with KDE. Instead of x_sample (50)
            # records, we pass this time only one
            pr += self._kernel_density_dict[class_value][feature].score_samples(
                np.array([row[feature]])[:, np.newaxis])[0]
          # The last operand is the number of times a record with that class
          # is given in the train dataset
          probability = pr + np.log(self._class_representation[class_value])
          s += probability
          # We save the class with a higher probability
          if probability > m:
            m, y = probability, class_value

        if s > -np.inf:
          y_pred.append(y)
        else:
          # If none of the classes has a probability greater than zero,
          # we assign the class that is most representative of the train dataset
          k = max(self._class_representation,
                  key=self._class_representation.get)  # type: ignore
          y_pred.append((self._class_representation[k]))

        next_bar()

    return np.array(y_pred)

  def _calculate_bandwidth(
      self,
      x: DataFrame,
      y: Series,
      bw_function_name: BWFunctionName,
      n_sample: int,
      class_values: set
  ) -> Dict[str, Dict[str, float]]:
    bw_dict = {}
    bw_function = get_bandwidth_function(bw_function_name)

    p_len = len(class_values) * len(x.columns)
    p_title = 'Calculating bandwidths'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for class_value in class_values:
        bw_dict[class_value] = {}
        d = x[y == class_value]
        for feature in x.columns:
          bw = bw_function(d[feature], n_sample)
          bw_dict[class_value][feature] = bw
          next_bar()

    return bw_dict

  def _calculate_kdes(
      self,
      x: DataFrame,
      y: Series,
      kernel: Kernel,
      algorithm: Algorithm,
      bw_dict: Dict[str, Dict[str, float]],
      n_sample: int,
      class_values: set
  ) -> List[KDE]:
    """Calculate the KDE for each class and feature."""
    kde_list = []

    p_len = len(class_values) * len(x.columns)
    p_title = 'Calculating all the KDEs'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for class_value in class_values:
        data_class = x[y == class_value]
        for feature in x.columns:
          # Calculate x_points
          data_var = x[feature]
          minimum, maximum = data_var.min(), data_var.max()
          x_points = np.linspace(minimum, maximum, n_sample)
          bw = bw_dict[class_value][feature]
          data = data_class[feature]

          # Fit Kernel Density
          kde = KernelDensity(
              kernel=kernel.value,
              bandwidth=bw,
              algorithm=algorithm.value
          ).fit(data.values[:, np.newaxis])

          # Calculate y_points
          y_points = np.exp(kde.score_samples(x_points[:, np.newaxis]))

          # Append the KDE object to the results list
          kde_list.append(KDE(
              feature,
              class_value,
              kde,
              x_points,
              y_points
          ))
          next_bar()

    return kde_list

  def _calculate_divergence(
      self,
      kde_list: List[KDE]
  ) -> DataFrame:
    """Calculate the divergence (distance) between each classes."""
    kde_dict = defaultdict(list[KDE])

    for kde in kde_list:
      kde_dict[kde.feature].append(kde)

    scores = []
    p_len = len(kde_dict)
    p_title = 'Calculating divergence'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for feature in kde_dict:
        for index_1 in range(len(kde_dict[feature]) - 1):
          t1 = kde_dict[feature][index_1].target_value
          f1 = kde_dict[feature][index_1].feature
          for index_2 in range(index_1 + 1, len(kde_dict[feature])):
            t2 = kde_dict[feature][index_2].target_value
            f2 = kde_dict[feature][index_2].feature
            if t1 != t2 and f1 == f2:
              p = self._normalize(kde_dict[feature][index_1].y_points)
              q = self._normalize(kde_dict[feature][index_2].y_points)
              hellinger = self._hellinger_distance(p, q)
              scores.append([f1, t1, t2, hellinger])
        next_bar()

    ranking = DataFrame(
        scores,
        columns=['feature', 'p0', 'p1', 'hellinger']
    ).drop_duplicates().sort_values(
        by=['hellinger', 'feature', 'p0', 'p1'],
        ascending=[False, True, True, True]
    )
    return ranking[ranking.hellinger > 0.5].reset_index(drop=True)

  @staticmethod
  def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Hellinger distance between two distributions."""
    try:
      s = sum([sqrt(a * b) for a, b in zip(p, q)])
    except ValueError:
      s = 0.0
    s = max(0, min(1, s))
    return sqrt(1 - s)

  @staticmethod
  def _normalize(data: np.ndarray) -> np.ndarray:
    s = np.sum(data)
    if s > 0:
      data = data / s
    s = np.sum(data)
    while s > 1.0:
      data = data / s
      s = np.sum(data)
    return data

  def _calculate_feature_selection(
          self,
          ranking: DataFrame,
          class_values: set
  ) -> Dict[str, Set[str]]:
    """Calculate the feature selection for each class."""
    threshold = 0.999
    stop_dict = defaultdict(dict[str, bool])
    hellinger_dict = defaultdict(set)
    for cv_1, cv_2 in product(class_values, repeat=2):
      if cv_1 != cv_2:
        stop_dict[cv_1][cv_2] = False

    p_len = len(ranking)
    p_title = 'Calculating feature selection'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for _, row in ranking.iterrows():
        feature = row.feature
        class_1 = row.p0
        class_2 = row.p1
        hellinger = row.hellinger
        hellinger_dict, stop_dict = self._update_feature_selection_dict(
            hellinger_dict,
            stop_dict,
            feature,
            hellinger,
            threshold,
            class_a=class_1,
            class_b=class_2
        )
        hellinger_dict, stop_dict = self._update_feature_selection_dict(
            hellinger_dict,
            stop_dict,
            feature,
            hellinger,
            threshold,
            class_a=class_2,
            class_b=class_1
        )
        next_bar()

    self._feature_selection_dict = defaultdict(set[str])
    for class_value in hellinger_dict.keys():
      self._feature_selection_dict[class_value] = {
          cfd.feature for cfd in hellinger_dict[class_value]
      }
    self._feature_selection_dict = dict(self._feature_selection_dict)
    return self._feature_selection_dict

  @staticmethod
  def _update_feature_selection_dict(
          hellinger_dict: Dict[str, Set[_ClassFeatureDistance]],
          stop_dict: Dict[str, Dict[str, bool]],
          feature: str,
          hellinger: float,
          threshold: float,
          class_a: str,
          class_b: str
  ) -> Tuple[
      Dict[str, Set[_ClassFeatureDistance]],
      Dict[str, Dict[str, bool]]
  ]:
    """Auxiliary method to calculate the feature selection."""
    if not stop_dict[class_a][class_b]:
      not_in_dict = class_a not in {
          x.class_value for x in hellinger_dict[class_b]
      }
      if not_in_dict:
        stop_dict[class_a][class_b] = hellinger >= threshold
      else:
        p = 1
        for x in hellinger_dict[class_b]:
          if x.class_value == class_a:
            p *= (1 - x.distance)
        stop_dict[class_a][class_b] = (1 - (p * (1 - hellinger))) >= threshold

      hellinger_dict[class_b].add(
          _ClassFeatureDistance(
              class_value=class_a,
              feature=feature,
              distance=hellinger
          )
      )
    return hellinger_dict, stop_dict

  def _calculate_target_representation(
      self,
      target_col: Series,
      class_values: Set
  ) -> Dict:
    """Calculate the percentage representation for each class."""
    self._class_representation = {}
    for class_value in class_values:
      target_count = target_col.value_counts().get(class_value, 0)
      total_count = len(target_col)
      self._class_representation[class_value] = target_count / total_count
    return self._class_representation

  def _calculate_necessary_kde(
      self,
      x: DataFrame,
      y: Series,
      bw_dict: Dict[str, Dict[str, float]],
      kernel: Kernel,
      algorithm: Algorithm
  ) -> Dict:
    """Calculate the KDE for each class."""
    self._kernel_density_dict = {}

    p_len = len(self._feature_selection_dict)
    p_title = 'Calculating only the necessary KDEs before the prediction'
    with progress_bar(self.show_progress_bar, p_len, p_title) as next_bar:
      for class_value, features in self._feature_selection_dict.items():
        data_class = x[y == class_value]
        self._kernel_density_dict[class_value] = {}
        for feature in features:
          data = data_class[feature]
          bw = bw_dict[class_value][feature]
          kde = KernelDensity(
              kernel=kernel.value,
              bandwidth=bw,
              algorithm=algorithm.value
          ).fit(data.values[:, np.newaxis])
          self._kernel_density_dict[class_value][feature] = kde
        next_bar()

    return self._kernel_density_dict


class NotFittedError(ValueError, AttributeError):
  """Exception class to raise if estimator is used before fitting.

  This class inherits from both ValueError and AttributeError to help with
  exception handling and backward compatibility.
  """

  def __init__(
      self,
      message=(
          'This XNB instance is not fitted yet.',
          ' Call "fit" with appropriate arguments before using this estimator.'
      )
  ):
    """Initialize the error."""
    self.message = message

  def __str__(self):
    """Return the message."""
    return self.message
