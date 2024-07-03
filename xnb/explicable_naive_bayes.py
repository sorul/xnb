from pandas import DataFrame, Series
from typing import Tuple, List, Dict, Callable, Set
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import sqrt
import numpy as np

from xnb._kde_object import KDE
from xnb.enum import Algorithm, BandwidthFunction, Kernel


__all__ = [
    'XNB'
]


class _ClassFeatureDistance:
  def __init__(self, class_value: str, feature: str, distance: float = 0.0):
    self.class_value = class_value
    self.feature = feature
    self.distance = distance

  def __hash__(self):
    return hash(self.class_value + self.feature)

  def __eq__(self, other):
    if isinstance(other, _ClassFeatureDistance):
      return self.class_value + self.feature == other.class_value + self.feature
    return False

  # def __str__(self) -> str:
  #   return f'{self.class_value} || {self.feature}'


class XNB:

  def __init__(self) -> None:
    pass

  def fit(
      self,
      x: DataFrame,
      y: Series,
      bw_function: BandwidthFunction = BandwidthFunction.HSCOTT,
      kernel: Kernel = Kernel.GAUSSIAN,
      algorithm: Algorithm = Algorithm.AUTO,
      n_sample: int = 50
  ) -> None:
    class_values = set(y)
    bw_dict = self._calculate_bandwidth(
        x, y, bw_function, n_sample, class_values)
    kde_list = self._calculate_kdes(
        x, y, kernel, algorithm, bw_dict, n_sample, class_values)
    ranking = self._calculate_divergence(kde_list)
    self._calculate_feature_selection(ranking, class_values)
    # self._calculate_necessary_kde(X, y)

    self._calculate_target_representation(y, class_values)

  def predict(self, x: DataFrame) -> Series:
    pass

  def _calculate_bandwidth(
      self,
      X: DataFrame,
      y: Series,
      bw_function: BandwidthFunction,
      n_sample: int,
      class_values: set
  ) -> Dict[str, Dict[str, float]]:
    bw_dict = {}

    for class_value in class_values:
      bw_dict[class_value] = {}
      d = X[y == class_value]
      for feature in X.columns:
        bw = bw_function.value(d[feature], n_sample)
        bw_dict[class_value][feature] = bw

    return bw_dict

  def _calculate_kdes(
      self,
      X: DataFrame,
      y: Series,
      kernel: Kernel,
      algorithm: Algorithm,
      bw_dict: Dict[str, Dict[str, float]],
      n_sample: int,
      class_values: set
  ) -> List[KDE]:
    kde_list = []

    for class_value in class_values:
      data_class = X[y == class_value]
      for feature in X.columns:
        # Calculate x_points
        data_var = X[feature]
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

    return kde_list

  def _calculate_divergence(
      self,
      kde_list: List[KDE]
  ) -> DataFrame:
    kde_dict = defaultdict(List[KDE])

    for kde in kde_list:
      kde_dict[kde.feature].append(kde)

    scores = []
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

    return DataFrame(
        scores,
        columns=['feature', 'p0', 'p1', 'hellinger']
    ).drop_duplicates().sort_values(
        by=['hellinger', 'feature', 'p0', 'p1'],
        ascending=False
    )

  @staticmethod
  def _hellinger_distance(p: List[float], q: List[float]) -> float:
    s = sum([sqrt(a * b) for a, b in zip(p, q)])
    s = max(0, min(1, s))
    return sqrt(1 - s)

  @staticmethod
  def _normalize(data: np.ndarray) -> List:
    s = sum(data)
    norm_data = [x / s if s != 0 else 0 for x in data]
    diff = 1 - sum(norm_data)
    norm_data[-1] += diff
    return norm_data

  # @staticmethod
  # def _normalize_2(data: List) -> List:
  #   s = sum(data)
  #   if s > 0:
  #     data = [data[i] / s for i in range(len(data))]
  #   else:
  #     data = [1.0 / len(data) for i in range(len(data))]
  #   s = sum(data)
  #   while s > 1.0:
  #     data = [data[i] / s for i in range(len(data))]
  #     s = sum(data)
  #   return data

  def _calculate_feature_selection(
          self,
          ranking: DataFrame,
          class_values: set
  ) -> None:
    threshold = 0.999
    stop_dict = defaultdict(dict[str, bool])
    hellinger_dict = defaultdict(dict)
    for cv_1, cv_2 in product(class_values, repeat=2):
      if cv_1 != cv_2:
        stop_dict[cv_1][cv_2] = False

    for _, row in ranking.iterrows():
      feature = row.feature
      class_1 = row.p0
      class_2 = row.p1
      hellinger = row.hellinger
      if hellinger > 0.5:
        hellinger_dict, stop_dict = self._calculate_feature_selection_dict(
            hellinger_dict,
            stop_dict,
            feature,
            hellinger,
            threshold,
            class_a=class_1,
            class_b=class_2
        )
        hellinger_dict, stop_dict = self._calculate_feature_selection_dict(
            hellinger_dict,
            stop_dict,
            feature,
            hellinger,
            threshold,
            class_a=class_2,
            class_b=class_1
        )

    # for d in dict_result:
    #   self.feature_selection_dict[d] = set(
    #       map(lambda x: x.split(' || ')[1], dict_result[d].keys()))

  @staticmethod
  def _calculate_feature_selection_dict(
          hellinger_dict: Dict[str, Set[_ClassFeatureDistance]],
          stop_dict: Dict[str, Dict[str, bool]],
          feature: str,
          hellinger: float,
          threshold: float,
          class_a: str,
          class_b: str
  ) -> Tuple[Dict, Dict]:
    """
    @param hellinger_dict:
    @param stop_dict:
    @param feature:
    @param hellinger:
    @param threshold:
    @param class_a:
    @param class_b:
    @return:
    """

    if not stop_dict[class_a][class_b]:
      hellinger_dict[class_b].add(
          _ClassFeatureDistance(
              class_value=class_a,
              feature=feature,
              distance=hellinger
          )
      )
      not_in_dict = class_a not in [
          x.class_value for x in hellinger_dict[class_b]
      ]
      if not_in_dict:
        stop_dict[class_a][class_b] = hellinger >= threshold
      else:
        p = 1
        for x in hellinger_dict[class_b]:
          if x.class_value == class_a:
            p *= (1 - x.distance)
        stop_dict[class_a][class_b] = (1 - (p * (1 - hellinger))) >= threshold

    return hellinger_dict, stop_dict

  def _calculate_target_representation(
      self,
      target_col: Series,
      class_values: Set
  ) -> Dict:
    class_representation = {}
    for class_value in class_values:
      target_count = target_col.count(class_value)
      total_count = len(target_col)
      class_representation[class_value] = target_count / total_count
    return class_representation
