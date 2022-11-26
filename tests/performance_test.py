from kde_classifier.stratified_naive_bayes import Stratified_NB
import pandas as pd
import sklearn.model_selection as model_selection
import time

# Pytest-Benchmark: https://pypi.org/project/pytest-benchmark/

def iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/iris.csv", sep=',')
    X, y = df.iloc[:, 0:-1], df.iloc[:,-1]
    return  X, y


def leukemia_dataset(n_cols=50) -> tuple[pd.DataFrame, pd.Series]:
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

'''
#######################
       	TESTS
#######################
'''

# > poetry run pytest -k kde --verbose

def test_kde_asyncio(benchmark):
    X, y = leukemia_dataset()
    snb = Stratified_NB()
    snb._class_values = set(y)
    snb._bw = getBandwidths(X, y)

    benchmark(snb._calculate_kde_asyncio, X, y)
    
    assert len(snb._kde_list) > 0 and len(snb._kernel_density_dict) > 0



# > poetry run pytest -k bandwidth --verbose

def test_calculate_bandwidth_best_estimator(benchmark):
    X, y = leukemia_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_BEST_ESTIMATOR)
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth, X, y)
    
    assert len(snb._bw) > 0

def test_calculate_bandwidth_hscott(benchmark):
    X, y = leukemia_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSCOTT)
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth, X, y)
    
    assert len(snb._bw) > 0

def test_calculate_bandwidth_hsilverman(benchmark):
    X, y = leukemia_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSILVERMAN)
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth, X, y)
    
    assert len(snb._bw) > 0

def test_calculate_bandwidth_hjs(benchmark):
    X, y = leukemia_dataset()
    snb = Stratified_NB(bw_function=Stratified_NB.BW_HSJ)
    snb._class_values = set(y)

    benchmark(snb._calculate_bandwidth, X, y)
    
    assert len(snb._bw) > 0

