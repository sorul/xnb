from kde_classifier.stratified_naive_bayes import Stratified_NB
from kde_classifier._kde_object import KDE
import pandas as pd
import itertools
import pytest
import time


# Pytest-Benchmark: https://pypi.org/project/pytest-benchmark/

def iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/iris.csv", sep=',')
    X, y = df.iloc[:, 0:-1], df.iloc[:,-1]
    return  X, y


def leukemia_dataset(n_cols=500) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/Leukemia_GSE9476.csv", sep=',').drop('samples', axis=1, errors='ignore')
    X, y = df.iloc[:, 1:n_cols], df.iloc[:,0]
    return X, y


def getBandwidths(X:pd.DataFrame, y:pd.Series) -> pd.DataFrame:
    variables = X.columns
    bw = {'variable':[], 'target':[], 'bandwidth':[]}
    for v in variables:
        for c in set(y):
            bw['variable'].append(v)
            bw['target'].append(c)
            bw['bandwidth'].append(0.1)
    return pd.DataFrame(bw)

def getKDES(X:pd.DataFrame, y:pd.Series) -> list[KDE]:
    comb = list(itertools.product(X.columns, list(set(y))))
    kde_list = []
    x_points = [i for i in range(10)]
    y_points = x_points
    for v, c in comb:
        kde_list.append(KDE(v, c, None, x_points, y_points))
    return kde_list

'''
#######################
       	TESTS
#######################
'''

# > poetry run pytest -k test_calculate_bandwidth_function -v

@pytest.mark.benchmark(
    min_time=0.1,
    max_time=0.5,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=False
)
def test_calculate_bandwidth_function(benchmark):
    X, y = iris_dataset()
    snb = Stratified_NB()
    snb._X = X
    snb._y = y
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth)
    
    assert len(snb._bw.variable.unique()) == len(X.columns) and len(snb._bw.target.unique()) == len(set(y))

# > poetry run pytest -k bandwidth -v

def test_calculate_bandwidth_best_estimator(benchmark):
    X, y = iris_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_BEST_ESTIMATOR)
    snb._X = X
    snb._y = y
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth)
    
    assert len(snb._bw.variable.unique()) == len(X.columns) and len(snb._bw.target.unique()) == len(set(y))

def test_calculate_bandwidth_hscott(benchmark):
    X, y = iris_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSCOTT)
    snb._X = X
    snb._y = y
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth)
    
    assert len(snb._bw.variable.unique()) == len(X.columns) and len(snb._bw.target.unique()) == len(set(y))

def test_calculate_bandwidth_hsilverman(benchmark):
    X, y = iris_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSILVERMAN)
    snb._X = X
    snb._y = y
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth)
    
    assert len(snb._bw.variable.unique()) == len(X.columns) and len(snb._bw.target.unique()) == len(set(y))

def test_calculate_bandwidth_hjs(benchmark):
    X, y = iris_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSJ)
    snb._X = X
    snb._y = y
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth)
    
    assert len(snb._bw.variable.unique()) == len(X.columns) and len(snb._bw.target.unique()) == len(set(y))

# > poetry run pytest -k kde -v

def test_calculate_kde(benchmark):
    X, y = leukemia_dataset(10)
    snb = Stratified_NB()
    snb._X = X
    snb._y = y
    snb._class_values = set(y)
    snb._bw = getBandwidths(X, y)

    benchmark(snb._calculate_kde)
    
    assert len(snb._kde_list) > 0


# > poetry run pytest -k test_calculate_divergence -v

def test_calculate_divergence(benchmark):
    X, y = leukemia_dataset(50)
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSJ)
    snb._X = X
    snb._y = y
    snb._class_values = set(y)
    snb._bw = getBandwidths(X, y)
    snb._kde_list = getKDES(X, y)

    benchmark(snb._calculate_divergence)

    assert len(snb._ranking_divergence) > 0
