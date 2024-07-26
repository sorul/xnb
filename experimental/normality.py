import os
import time
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import norm
import itertools
from scipy.stats import entropy
from numba import jit, vectorize, cuda
from pathlib import Path
import cProfile
import pstats
from typing import Tuple
from collections import defaultdict
from line_profiler import LineProfiler

from experimental.independence import (
    independence_test,
    independence_test_4,
    independence_test_3,
    calculate_partial_corr_matrix
)

# Dictionary to store statistical tests
statistics = {
    'shapiro_wilk': shapiro
}


def shapiro_wilk(X, y):
  count = 0
  for variable in X.columns:
    result = statistics['shapiro_wilk'](X[variable])
    if result.pvalue < 0.05:
      count = count + 1
  return count


def conditional_mutual_information(X, y):
  x3 = y
  count = 0

  # Create dummy variables for the categorical variable x3
  x3_dummies = np.column_stack([x3 == category for category in np.unique(x3)])
  x3 = x3_dummies

  for i in range(0, len(X.columns)-1):
    for j in range(i+1, len(X.columns)):
      # Compute the residuals of x1 and x2 after regressing out the effect of x3
      x1 = X.iloc[:, i]
      x2 = X.iloc[:, j]

      # Compute joint entropy H(X,Y|Z)
      joint_xyz = np.histogram2d(x1, x2, bins=(
          len(np.unique(x1)), len(np.unique(x2))), density=True)[0]
      joint_z = np.histogram(x3, bins=len(np.unique(x3)), density=True)[0]
      H_XY_Z = entropy(joint_xyz.flatten()) - entropy(joint_z)

      # Compute conditional entropy H(X|Y,Z)
      H_X_YZ = entropy(joint_xyz.flatten(), axis=0) - entropy(joint_z)

      # Compute conditional mutual information I(X;Y|Z)
      cmi = H_XY_Z - H_X_YZ

      if cmi > 0.95:  # If close to 0 suggets X and Y are conditionally dependent given Z.
        count = count + 1
        break
    # if i%10 == 0:
    #    print(i)
  return count


def load_dataset(
        file_path: Path,
        class_column: str = 'class',
        n_cols: int = 1000,
        sep: str = ';'
) -> Tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv(
      file_path, sep=sep
  ).drop('samples', axis=1, errors='ignore').head(1000)
  y = df[class_column]
  x = df.drop(class_column, axis=1)
  x = x[list(x.columns)[:n_cols]]
  return x, y


def iterar_datasets():
  analysis_results_dict = defaultdict(list)
  dataset_names = {
      # 'iris.csv': 'class',
      'GSE42568.csv': 'ID_REF'
  }
  for dataset_name, class_name in dataset_names.items():
    x, y = load_dataset(Path(f'data/{dataset_name}'), class_column=class_name)

    count_normal = shapiro_wilk(x, y)
    result_normal = count_normal / len(x.columns)
    analysis_results_dict[dataset_name].append(round(result_normal, 3))
    count_independence = independence_test_4(x, y)
    result_independence = count_independence / (len(x.columns))
    analysis_results_dict[dataset_name].append(round(result_independence, 3))
    analysis_results_dict[dataset_name].append(len(x))
    analysis_results_dict[dataset_name].append(len(x.columns))
    analysis_results_dict[dataset_name].append(len(y.unique()))


if __name__ == "__main__":

  # Create a LineProfiler object
  profiler = LineProfiler()

  # # Add the functions you want to profile
  # profiler.add_function(calculate_partial_corr_matrix.__wrapped__)

  # Run the profiler on the test_main_normality function
  profiler_wrapper = profiler(iterar_datasets)
  profiler_wrapper()

  # Print the profiling results
  profiler.print_stats()
