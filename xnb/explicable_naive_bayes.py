"""Explicable Naive Bayes."""
from typing import Tuple, List, Dict, Set, Union, ClassVar
from pandas import DataFrame, Series
from collections import defaultdict
from itertools import product
from math import sqrt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin

from xnb._kde_object import KDE
from xnb.enums import Algorithm, BWFunctionName, Kernel
from xnb._bandwidth_functions import get_bandwidth_function
from xnb._progress_bar import progress_bar

__all__ = ['XNB', 'NotFittedError']


class _ClassFeatureDistance:
  def __init__(
      self,
      class_value: str,
      feature: str,
      distance: float = 0.0,
  ):
    self.class_value = class_value
    self.feature = feature
    self.distance = distance

  def __hash__(self):
    return hash(str(self.class_value) + str(self.feature))


class XNB(ClassifierMixin, BaseEstimator):
  """Class to perform Explainable Naive Bayes."""

  _estimator_type: ClassVar[str] = 'classifier'

  def __init__(
      self,
      bw_function: Union[BWFunctionName, str] = BWFunctionName.HSILVERMAN,
      kernel: Union[Kernel, str] = Kernel.GAUSSIAN,
      algorithm: Union[Algorithm, str] = Algorithm.AUTO,
      n_sample: int = 50,
      show_progress_bar: bool = False,
  ) -> None:
    """Initialize the Explainable Naive Bayes model.

    Args:
        bw_function (Union[BWFunctionName, str], optional): Bandwidth selection
          function. Defaults to BWFunctionName.HSILVERMAN.
        kernel (Union[Kernel, str], optional): Kernel type for density
          estimation. Defaults to Kernel.GAUSSIAN.
        algorithm (Union[Algorithm, str], optional): Algorithm for kernel
          density estimation. Defaults to Algorithm.AUTO.
        n_sample (int, optional): Number of samples for density estimation.
          Defaults to 50.
        show_progress_bar (bool, optional): Whether to display a progress bar
          during fitting. Defaults to False.
    """
    self.bw_function = str(bw_function)
    self.kernel = str(kernel)
    self.algorithm = str(algorithm)
    self.n_sample = n_sample
    self.show_progress_bar = show_progress_bar

  @property
  def feature_selection_dict(
      self
  ) -> Dict[Union[str, float], Set[Union[str, float]]]:
    """Obtain the feature selection dictionary.

    Each target class (key) is associated with a set of features (values).
    """
    if hasattr(self, '_feature_selection_dict'):
      return self._feature_selection_dict
    else:
      raise NotFittedError()

  def _repr_html_(self):  # pragma: no cover
    """Return html representation of the model."""
    return (
        '''
      <style>
          .sklearn-mime-model {{
              display: inline-block;
              padding: 8px 12px;
              background: #f8f9fa;
              border-left: 5px solid #007bff;
              border-radius: 4px;
              font-family: monospace;
              font-size: 14px;
          }}  # noqa
      </style>
    '''
    )  # noqa

  def fit(
      self,
      x: DataFrame,
      y: Series,
  ) -> 'XNB':
    """Calculate the best feature selection to be able to predict later.

    Args:
        x (DataFrame):  DataFrame containing the input features
        y (Series): Series containing the target variable

    Returns:
        XNB: Returns the instance itself.
    """
    class_values = set(y)
    bw_dict = self._calculate_bandwidth(
        x, y, self.bw_function, self.n_sample, class_values
    )
    kde_list = self._calculate_kdes(
        x, y, self.kernel, self.algorithm, bw_dict, self.n_sample, class_values
    )
    ranking = self._calculate_divergence(kde_list)
    self._calculate_feature_selection(ranking, class_values)
    self._calculate_necessary_kde(x, y, bw_dict, self.kernel, self.algorithm)
    self._calculate_target_representation(y, class_values)

    # scikit-learn compatibility fields
    self.classes_ = np.array(sorted(class_values))

    self.is_fitted_ = True
    return self

  def predict_proba(self, x: DataFrame) -> np.ndarray:
    """Return the probabilities of each class for all rows in the DataFrame.

    Args:
        x (DataFrame): DataFrame containing the input to predict probabilities.

    Returns:
        np.ndarray: Array where each row contains the probabilities for each
        class.
    """
    self._check_if_not_fitted_error()

    log_probs = self._calculate_class_log_probabilities(x)
    return self._normalize_probabilities(log_probs)

  def predict(self, x: DataFrame) -> np.ndarray:
    """Return the predicted class for each row in the DataFrame.

    Args:
        x (DataFrame): DataFrame containing the input to predict.

    Returns:
        np.ndarray: Numpy array containing the predicted class for each row.
    """
    self._check_if_not_fitted_error()

    log_probs = self._calculate_class_log_probabilities(x)
    probabilities = self.predict_proba(x)
    class_indices = np.argmax(probabilities, axis=1)
    return np.array(sorted(log_probs.keys()))[class_indices]

  def _check_if_not_fitted_error(self) -> None:
    cond1 = not hasattr(self, '_kernel_density_dict')
    cond2 = not hasattr(self, '_class_representation')
    cond3 = not hasattr(self, 'is_fitted_')

    if cond1 or cond2 or cond3:
      raise NotFittedError()

  def _calculate_bandwidth(
      self,
      x: DataFrame,
      y: Series,
      bw_function_name: str,
      n_sample: int,
      class_values: set,
  ) -> Dict[Union[str, float], Dict[Union[str, float], float]]:
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
      kernel: str,
      algorithm: str,
      bw_dict: Dict[Union[str, float], Dict[Union[str, float], float]],
      n_sample: int,
      class_values: set,
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
              kernel=kernel,  # type: ignore
              bandwidth=bw,
              algorithm=algorithm,  # type: ignore
          ).fit(data.values[:, np.newaxis])

          # Calculate y_points
          y_points = np.exp(kde.score_samples(x_points[:, np.newaxis]))

          # Append the KDE object to the results list
          kde_list.append(KDE(feature, class_value, kde, x_points, y_points))
          next_bar()

    return kde_list

  def _calculate_divergence(self, kde_list: List[KDE]) -> DataFrame:
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
        columns=['feature', 'p0', 'p1', 'hellinger'],
    ).drop_duplicates().sort_values(
        by=['hellinger', 'feature', 'p0', 'p1'],
        ascending=[False, True, True, True],
    )
    threshold = sum(ranking.hellinger) / len(ranking.hellinger)
    return ranking[ranking.hellinger > threshold].reset_index(drop=True)

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
      class_values: set,
  ) -> Dict[str, Set[Union[str, float]]]:
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

    self._feature_selection_dict = defaultdict(set[Union[str, float]])
    for class_value in hellinger_dict.keys():
      cv = (
          float(class_value) if isinstance(class_value, float) else class_value
      )
      self._feature_selection_dict[cv] = {
          float(cfd.feature) if isinstance(cfd.feature, float) else cfd.feature
          for cfd in hellinger_dict[cv]
      }
    self._feature_selection_dict = dict(self._feature_selection_dict)
    return self._feature_selection_dict

  @staticmethod
  def _update_feature_selection_dict(
      hellinger_dict: Dict[Union[str, float], Set[_ClassFeatureDistance]],
      stop_dict: Dict[str, Dict[str, bool]],
      feature: str,
      hellinger: float,
      threshold: float,
      class_a: str,
      class_b: str,
  ) -> Tuple[
      Dict[Union[str, float], Set[_ClassFeatureDistance]],
      Dict[str, Dict[str, bool]],
  ]:
    """Auxiliary method to calculate the feature selection."""
    if not stop_dict[class_a][class_b]:
      not_in_dict = class_a not in {
          x.class_value
          for x in hellinger_dict[class_b]
      }
      if not_in_dict:
        stop_dict[class_a][class_b] = hellinger >= threshold
      else:
        p = 1
        for x in hellinger_dict[class_b]:
          if x.class_value == class_a:
            p *= max(1 - x.distance, 1e-6)
        stop_dict[class_a][class_b] = (1 - (p * (1 - hellinger))) >= threshold

      hellinger_dict[class_b].add(
          _ClassFeatureDistance(
              class_value=class_a, feature=feature, distance=hellinger
          )
      )
    return hellinger_dict, stop_dict

  def _calculate_target_representation(
      self,
      target_col: Series,
      class_values: Set,
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
      bw_dict: Dict[Union[str, float], Dict[Union[str, float], float]],
      kernel: str,
      algorithm: str,
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
              kernel=kernel,  # type: ignore
              bandwidth=bw,
              algorithm=algorithm,  # type: ignore
          ).fit(data.values[:, np.newaxis])
          self._kernel_density_dict[class_value][feature] = kde
        next_bar()

    return self._kernel_density_dict

  def _calculate_class_log_probabilities(self, x: DataFrame) -> Dict:
    """Calculate the log probabilities for all classes for multiple rows.

    ## Args:
    :param x: DataFrame containing the input data.

    ## Returns:
    :return: Dictionary where each class is a key
    and the value is an array of log probabilities.
    """
    log_probs = {
        class_value: np.zeros(len(x))
        for class_value in self.feature_selection_dict
    }

    for class_value, features in self.feature_selection_dict.items():
      for feature in features:
        # Get KDE scores for the entire column
        log_probs[class_value] += self\
            ._kernel_density_dict[class_value][feature].score_samples(
            x[feature].to_numpy()[:, np.newaxis]
        )
      # Add log prior probabilities
      log_probs[class_value] += np.log(self._class_representation[class_value])

    return log_probs

  def _normalize_probabilities(self, log_probs: Dict) -> np.ndarray:
    """Normalize log probabilities to return actual probabilities for all rows.

    ## Args:
    :param log_probs: Dictionary of log probabilities for each class.

    ## Returns:
    :return: Array of shape (n_samples, n_classes) with normalized probs.
    """
    log_probs_matrix = np.vstack(
        [log_probs[class_value] for class_value in sorted(log_probs.keys())]
    ).T
    probs_matrix = np.exp(log_probs_matrix)
    probs_matrix /= probs_matrix.sum(axis=1, keepdims=True)
    return probs_matrix


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
