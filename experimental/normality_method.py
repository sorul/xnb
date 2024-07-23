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


def shapiro_wilk(X, y):
  count = 0
  for variable in X.columns:
    result = statistics['shapiro_wilk'](X[variable])
    if result.pvalue < 0.05:
      count = count + 1
  return count


def pearson(x1, x2, x3_dummies):
  # Compute the residuals of x1 and x2 after regressing out the effect of x3
  x1_res = x1 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x1, rcond=None)[0])
  x2_res = x2 - \
      np.dot(x3_dummies, np.linalg.lstsq(x3_dummies, x2, rcond=None)[0])

  # Compute the partial correlation coefficient between x1 and x2 controlling for x3
  partial_corr, p_value = pearsonr(x1_res, x2_res)
  return partial_corr, p_value


def calculate_dependency(X, pair, x3_dummies):
  global cont_ind_vars
  global cont_total
  dependent = []
  x1 = X[pair[0]]
  x2 = X[pair[1]]
  cont_total = cont_total + 1
  partial_corr, p_value = pearson(x1, x2, x3_dummies)
  # If <0.05, reject the null hypothesis that the correlation coefficient is zero, indicating a significant correlation.
  if p_value < 1e-6 and abs(partial_corr) > 0.7:
    cont_ind_vars = cont_ind_vars + 1
    print(pair[0], pair[1], partial_corr, p_value, cont_total, cont_ind_vars)
    dependent.append(pair[0])
    dependent.append(pair[1])
  return dependent


def independence_test2(X, y):

  ind_vars = {}
  # Create dummy variables for the categorical variable x3
  x3_dummies = np.column_stack([y == category for category in np.unique(y)])
  pairs = list(itertools.combinations(X.columns, 2))
  print(len(pairs))
  dependent = list(
      map(lambda pair: calculate_dependency(X, pair, x3_dummies), pairs))
  # dependent = [x for x in dependent if x]
  # ind_vars = list(map(lambda pair: ))
  for pair in dependent:
    if pair != []:
      ind_vars[pair[0]] = 1
      ind_vars[pair[1]] = 1
  return len(ind_vars)


def independence_test(X, y):
  x3 = y
  count = 0
  cont_total = 0

  # Create dummy variables for the categorical variable x3
  x3_dummies = np.column_stack([x3 == category for category in np.unique(x3)])

  dependent = [0] * len(X.columns)

  for i in range(0, len(X.columns)-1):
    #    if (dependent[i]==0):
    for j in range(i+1, len(X.columns)):
      if not (dependent[i] == 1 and dependent[j] == 1):
        # Compute the residuals of x1 and x2 after regressing out the effect of x3
        x1 = X.iloc[:, i]
        x2 = X.iloc[:, j]

        partial_corr, p_value = pearson(x1, x2, x3_dummies)
        cont_total = cont_total + 1
        # print(partial_corr, p_value)
        # If <0.05, reject the null hypothesis that the correlation coefficient is zero, indicating a significant correlation.
        if p_value < 1e-6 and abs(partial_corr) > 0.7:
          count = count + 1
          print(i, j, partial_corr, p_value, cont_total, count)
          dependent[i] = 1
          dependent[j] = 1
          # break
    # if i%100 == 0:
    #    print(i)
  print(sum(dependent), count)
  return sum(dependent)


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


if __name__ == "__main__":
  cont_ind_vars = 0
  cont_total = 0

  # Data structures for storing analysis results
  analysis_results_dict = {}
  analysis_columns = ['Dataset']

  # Parameters
  directory_path = "/Users/aguilar/Documents/DATOS/PAPERS/TALLER/2024/++EXPLAINABLE NAIVE BAYES/ALLDATA"
  class_col = "type"

  # Retrieving CSV files from 'Datasets/' directory
  datasets_dir = os.listdir(directory_path)
  datasets_dir = list(filter(lambda str: str.endswith('.csv'), datasets_dir))
  datasets_dir.sort()

  # Dictionary to store statistical tests
  statistics = {
      'shapiro_wilk': shapiro
  }

  # Looping through each dataset
  for dataset_name in datasets_dir:
    dataset_start_time = time.time()
    # Extracting training and testing data
    dataset_path = directory_path + "/" + dataset_name
    dataset = pd.read_csv(f'{dataset_path}', sep=',')

    print(f'{dataset_name}')

    # print(dataset.isnull().sum())

    # If there's no labeled class column, continue
    if class_col not in dataset.columns:
      continue

    # Genomic datasets use to include a column named "samples", which stands for sample identifier (not relevant for classification)
    dataset = dataset.drop('samples', errors='ignore', axis=1)
    X = dataset.drop(columns=class_col)
    # Some datasets contain NaN values and classifiers produce errors when trying to fit a model
    # X = aux_prepro.one_hot_encoder(X.fillna(X.mean()))
    y = dataset[class_col]

    # Creating List for each dataset name
    analysis_results_dict[dataset_name] = []

    count_normal = shapiro_wilk(X, y)
    # Ratio of variables that do not follow normal distribution
    result_normal = count_normal / len(X.columns)
    analysis_results_dict[dataset_name].append(round(result_normal, 3))

    count_independence = independence_test(X, y)
    # Ratio of variable pairs that are conditionally dependent.
    result_independence = count_independence / (len(X.columns))
    analysis_results_dict[dataset_name].append(round(result_independence, 3))
    # analysis_results_dict[dataset_name].append(0.000)

    # count_independence = conditional_mutual_information(X,y)
    # result_independence = count_independence / (len(X.columns)) # Ratio of variable that are conditionally dependent.
    # analysis_results_dict[dataset_name].append(round(result_independence,3))

    analysis_results_dict[dataset_name].append(len(X))
    analysis_results_dict[dataset_name].append(len(X.columns))
    analysis_results_dict[dataset_name].append(len(y.unique()))

    dataset_end_time = time.time()
    dataset_total_time = dataset_end_time - dataset_start_time
    print(f'Total time ({dataset_name}) : {dataset_total_time:.3f} sec.')

  # Creating DataFrame for analysis results
  analysis_columns.append('Shapiro_wilk')
  analysis_columns.append('Pearson')
  analysis_columns.append('Instances')
  analysis_columns.append('Variables')
  analysis_columns.append('Classes')
  analysis_results = pd.DataFrame(columns=analysis_columns)
  for dataset_name in analysis_results_dict:
    analysis_results.loc[len(analysis_results.index)] = [
        dataset_name] + analysis_results_dict[dataset_name]

  # Saving analysis results to CSV file
  analysis_results.set_index('Dataset')
  analysis_results.to_csv('Analysis_results.csv',
                          sep=';', decimal=',', index=False)
