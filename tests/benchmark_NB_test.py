from kde_classifier.stratified_naive_bayes import Stratified_NB
from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
import pandas as pd


def iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/iris.csv", sep=',')
    X, y = df.iloc[:, 0:-1], df.iloc[:,-1]
    return  X, y


def leukemia_dataset(n_cols = 500) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/Leukemia_GSE9476.csv", sep=',').drop('samples', axis=1, errors='ignore')
    X, y = df.iloc[:, 1:n_cols], df.iloc[:,0]
    return X, y

'''
#######################
       	TESTS
#######################
'''

# > poetry run pytest -k benchmark --verbose

def test_accuracy_benchmark_naive_bayes():
    X, y = iris_dataset()
    accuracy_list = {'snb':[], 'nb1':[], 'nb2':[]}
    n_features_selected = []
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle = True, random_state=0)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # STRATIFIED NB
        snb = Stratified_NB()
        snb.fit(X_train, y_train)
        feature_selection = snb.feature_selection_dict
        y_pred = snb.predict(X_test)
        accuracy_list['snb'].append(accuracy_score(y_test, y_pred))

        # ORIGINAL NB
        nb = GaussianNB()
        nb.fit(X_train,y_train)
        y_pred = nb.predict(X_test)
        accuracy_list['nb1'].append(accuracy_score(y_test, y_pred))

        # FEATURE SELECTION NB
        new_cols = list({x for v in feature_selection.values() for x in v})
        nb = GaussianNB()
        nb.fit(X_train[new_cols],y_train)
        y_pred = nb.predict(X_test[new_cols])
        accuracy_list['nb2'].append(accuracy_score(y_test, y_pred))
        n_features_selected.append(len(new_cols))

    results = pd.DataFrame({'Accuracy SNB':accuracy_list['snb'], 'Accuracy NB1':accuracy_list['nb1'], 'Accuracy NB2':accuracy_list['nb2'], '#VAR':n_features_selected})
    snb_mean = results['Accuracy SNB'].mean()
    nb1_mean = results['Accuracy NB1'].mean()
    nb2_mean = results['Accuracy NB2'].mean()
    n_features_mean = results['#VAR'].mean()
    print("SBN: "+str(snb_mean))
    print("NB1: "+str(nb1_mean))
    print("NB2: "+str(nb2_mean))
    print("#VAR: "+str(n_features_mean))
    
    assert snb_mean >= nb1_mean-nb1_mean*0.05 and snb_mean >= nb2_mean-nb2_mean*0.05# and n_features_mean<len(X.columns)

