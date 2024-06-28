import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
# import multiprocessing
# import multiprocessing.managers
# from multiprocessing import Pool
import mpmath as mp
import itertools
from math import log2, prod, ceil, log10, log, sqrt
from xnb import _bandwidth_functions as bf
from xnb_jesus._kde_object import KDE
import asyncio
import time
from progress.bar import Bar, ChargingBar
from sklearn import preprocessing
# from KDEpy import FFTKDE
# import matplotlib.pyplot as plt

offset_print = 80


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  RED = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


class XNB():

  # Bandwitdh functions
  BW_HSILVERMAN = "hsilverman"
  BW_HSCOTT = "hscott"
  BW_HSJ = "hsj"
  # BW_HSJ = "improved_sheather_jones"
  BW_BEST_ESTIMATOR = "best_estimator"

  # Kernels
  # kernel{‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}, default=’gaussian’
  K_GAUSSIAN = "gaussian"
  K_COSINE = "cosine"
  K_EPANECHNIKOV = "epanechnikov"
  K_TOPHAT = "tophat"
  K_EXPONENTIAL = "exponential"
  K_LINEAR = "linear"

  # Algorithm
  # algorithm{‘kd_tree’, ‘ball_tree’, ‘auto’}, default=’auto’
  KD_TREE = "kd_tree"
  BALL_TREE = "ball_tree"

  def __init__(self, kernel: str = K_GAUSSIAN, margin_percentage: float = 0, x_sample: int = 50, bw_function: str = BW_HSCOTT, algorithm: str = BALL_TREE) -> None:
    # Public
    self.kernel = kernel
    self.algorithm = algorithm
    self.margin_percentage = margin_percentage
    self.x_sample = x_sample
    self.bw_function = bw_function
    self.feature_selection_dict = {}

    # Private
    self._kde_list: list[KDE]
    self._kernel_density_dict: dict[KernelDensity]
    self._ranking_divergence: pd.DataFrame
    self._class_values: set
    self._bw: pd.DataFrame
    self._class_representation: dict
    self._bw_list: list
    self._bw_dict: dict
    self._variables: list

  def __background(f):
    def wrapped(*args, **kwargs):
      return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

  def _calculate_bandwidth(self, X: pd.DataFrame, y: pd.Series) -> None:
    inicio = time.time()
    progressBar = Bar('PROG. BANDWIDTH:', max=len(
        X.columns)*len(self._class_values))
    # Different types of function can be used
    if self.bw_function == self.BW_HSILVERMAN:
      bw_f = bf.hsilverman
    elif self.bw_function == self.BW_HSCOTT:
      bw_f = bf.hscott
    elif self.bw_function == self.BW_HSJ:
      bw_f = bf.hsj
    elif self.bw_function == self.BW_BEST_ESTIMATOR:
      bw_f = bf.best_estimator
    else:
      raise ValueError("'"+self.bw_function +
                       "' is not a valid value for a bandwidth function.")

    self._bw_dict = {}
    for c in self._class_values:
      self._bw_dict[c] = {}
      d = X[y == c]
      for v in X.columns:
        bw = bw_f(d[v], self.x_sample)
        self._bw_dict[c][v] = bw
        progressBar.next()

    fin = time.time()
    progressBar.finish()
    print(f'T. BANDWIDTH: {fin-inicio:.3f} sec.'.rjust(offset_print))

  def _calculate_kde(self, X: pd.DataFrame, y: pd.Series) -> None:
    inicio = time.time()
    progressBar = Bar('PROG. KDE:', max=len(X.columns)*len(self._class_values))
    self._kde_list = []
    cols = X.columns
    mp.dps = 100

    for c in self._class_values:
      # self._kernel_density_dict[c] = {}
      data_class = X[y == c]
      for v in cols:
        data_var = X[v]
        minimum, maximum = data_var.min(), data_var.max()
        data = data_class[v]
        x_points = np.linspace(minimum, maximum, self.x_sample)
        bw = self._bw_dict[c][v]
        kde = KernelDensity(kernel=self.kernel, bandwidth=bw,
                            algorithm=self.algorithm).fit(data.values[:, np.newaxis])
        y_points = np.exp(kde.score_samples(x_points[:, np.newaxis]))
        self._kde_list.append(KDE(v, c, x_points, y_points))
        progressBar.next()
    fin = time.time()
    progressBar.finish()
    print(f'T. KDE: {fin-inicio:.3f} sec.'.rjust(offset_print))

  def _normalize(self, data: list) -> list:
    # Code has been changed to assure that sum(data)=1, as sometimes sum(data) was slightly over 1, e.g. 1.0000000002, and then the Hellinger distance did not work because 1-s<0.
    s = sum(data)
    if s > 0:
      data = [data[i] / s for i in range(len(data))]
    else:
      data = [1.0/len(data) for i in range(len(data))]
    s = sum(data)
    while s > 1.0:
      data = [data[i] / s for i in range(len(data))]
      s = sum(data)
    return data

  def _calculate_divergence(self) -> None:
    inicio = time.time()

    def hellinger_distance(p: list, q: list):
      s = sum([sqrt(a*b) for a, b in zip(p, q)])
      if s < 0:
        print("\n Hellinger distance < 0: ", s)
        s = 0
      elif s > 1:
        print("\n Hellinger distance > 0: ", s)
        s = 1

      return sqrt(1-s)

    progressBar = Bar('PROG. HELLINGER:', max=len(self._variables))

    kde_dict = {}
    for v in self._variables:
      kde_dict[v] = []

    for kde in self._kde_list:
      kde_dict[kde.feature].append(kde)

    scores = []
    for c in kde_dict:
      for kde1 in range(len(kde_dict[c])-1):
        t1, f1 = kde_dict[c][kde1].target, kde_dict[c][kde1].feature
        for kde2 in range(kde1+1, len(kde_dict[c])):
          t2, f2 = kde_dict[c][kde2].target, kde_dict[c][kde2].feature
          if t1 != t2 and f1 == f2:
            p = self._normalize(kde_dict[c][kde1].y_points)
            q = self._normalize(kde_dict[c][kde2].y_points)
            hellinger = hellinger_distance(p, q)
            scores.append([f1, t1, t2, hellinger])
      progressBar.next()

    ranking = pd.DataFrame(
        scores, columns=['variable', 'p0', 'p1', 'hellinger']).drop_duplicates()
    ranking = ranking.sort_values(
        by=['hellinger', 'variable', 'p0', 'p1'], ascending=False)
    ranking.to_csv("data/ranking.csv")

    self._ranking_divergence = ranking
    fin = time.time()
    progressBar.finish()
    print(f'T. HELLINGER: {fin-inicio:.3f} sec.'.rjust(offset_print))

  def _calculate_feature_selection(self):
    inicio = time.time()
    progressBar = Bar('PROG. FEATURE SELECTION:',
                      max=len(self._ranking_divergence))
    threshold = 0.999
    finished_class, dict_result = {}, {}
    for c in self._class_values:
      dict_result[c], finished_class[c] = {}, {}
      for c2 in self._class_values:
        if (c != c2):
          finished_class[c][c2] = False

    def addDict(dict_result, class_1, class_2, variable, hellinger, finished_class):
      if not finished_class[class_1][class_2]:
        k = class_1+' || '+variable
        not_in_dict = class_1 not in set(
            map(lambda x: x.split(' || ')[0], dict_result[class_2].keys()))
        if not_in_dict:
          dict_result[class_2][k] = hellinger
          finished_class[class_1][class_2] = hellinger >= threshold
        else:
          class_list = list(map(lambda x: x.split(
              ' || ')[0], dict_result[class_2].keys()))
          p = 1
          for i in range(len(class_list)):
            if (class_list[i] == class_1):
              valor = list(dict_result[class_2].values())[i]
              p *= (1-valor)
          dict_result[class_2][k] = hellinger
          finished_class[class_1][class_2] = (1-(p*(1-hellinger))) >= threshold
      return dict_result

    for _, row in self._ranking_divergence.iterrows():
      variable = row.variable
      class_1 = row.p0
      class_2 = row.p1
      hellinger = row.hellinger
      if hellinger > 0.5:
        dict_result = addDict(dict_result, class_1, class_2,
                              variable, hellinger, finished_class)
        dict_result = addDict(dict_result, class_2, class_1,
                              variable, hellinger, finished_class)
      progressBar.next()

    for d in dict_result:
      self.feature_selection_dict[d] = set(
          map(lambda x: x.split(' || ')[1], dict_result[d].keys()))
    fin = time.time()
    progressBar.finish()
    print(f'T. FEATURE SELECTION: {fin-inicio:.3f} sec.'.rjust(offset_print))

  def _calculate_target_representation(self, y: pd.Series) -> None:
    self._class_representation = {}
    for target in self._class_values:
      self._class_representation[target] = len(
          [i for i in y if i == target]) / len(y)

  def _calculate_necessary_kde(self, X: pd.DataFrame, y: pd.Series) -> None:
    inicio = time.time()
    progressBar = Bar('PROG. NECESSARY KDE:',
                      max=len(self.feature_selection_dict))
    self._kernel_density_dict = {}
    for c, variables in self.feature_selection_dict.items():
        # kde_values = []
      data_class = X[y == c]
      self._kernel_density_dict[c] = {}
      for v in variables:
        data_var = X[v]
        minimum, maximum = data_var.min(), data_var.max()
        data = data_class[v]
        x_points = np.linspace(minimum, maximum, self.x_sample)
        bw = self._bw_dict[c][v]
        kde = KernelDensity(kernel=self.kernel, bandwidth=bw,
                            algorithm=self.algorithm).fit(data.values[:, np.newaxis])
        self._kernel_density_dict[c][v] = kde

      progressBar.next()

    fin = time.time()
    progressBar.finish()
    print(f'T. NECESSARY KDE: {fin-inicio:.3f} sec.'.rjust(offset_print))

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
    self._class_values = set(y)
    self._variables = X.columns
    self._calculate_target_representation(y)
    self._calculate_bandwidth(X, y)
    self._calculate_kde(X, y)
    self._calculate_divergence()
    self._calculate_feature_selection()
    self._calculate_necessary_kde(X, y)

  def predict(self, X: pd.DataFrame) -> list:
    inicio = time.time()
    progressBar = Bar('PROG. PREDICT NEW:', max=len(X))
    # mp.dps = 1000

    # Iterating each test record
    y_pred = []
    for _, row in X.iterrows():
      # Calculating the probability of each class
      y, m, s = None, -np.inf, 0
      # Running through the final variables
      for c, variables in self.feature_selection_dict.items():
        kde_values = []
        pr = 0
        for v in variables:
          # We get the probabilities with KDE. Instead of x_sample (50) records, we pass this time only one
          k = self._kernel_density_dict[c][v].score_samples(
              np.array([row[v]])[:, np.newaxis])[0]
          pr = pr + k
        # The last operand is the number of times a record with that class is given in the train dataset
        probability = pr + np.log(self._class_representation[c])
        s += probability
        # We save the class with a higher probability
        if probability > m:
          m, y = probability, c

      if s > -np.inf:
        y_pred.append(y)
      else:
        # If none of the classes has a probability greater than zero, we assign the class that is most representative of the train dataset
        k = max(self._class_representation, key=self._class_representation.get)
        y_pred.append((self._class_representation[k]))
      progressBar.next()
    fin = time.time()
    progressBar.finish()
    print(f'T. PREDICTION NEW: {fin-inicio:.3f} sec.'.rjust(offset_print))
    return y_pred
