import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import multiprocessing
import multiprocessing.managers
from multiprocessing import Pool
import mpmath as mp
import itertools
from math import log2, prod, ceil, log10, sqrt
from kde_classifier import hselect
from kde_classifier.kde_object import KDE
import asyncio

class Stratified_NB():

    # Bandwitdh functions
    BW_HSILVERMAN = "hsilverman"
    BW_HSCOTT = "hscott"
    BW_HSJ = "hsj"
    BW_BEST_ESTIMATOR = "best_estimator"

    # Kernels
    K_GAUSSIAN = "gaussian"

    def __init__(self, kernel:str=K_GAUSSIAN, margin_percentage:float=0, x_sample:int=50, bw_function:str=BW_HSILVERMAN) -> None:
        # Public
        self.kernel = kernel
        self.margin_percentage = margin_percentage
        self.x_sample = x_sample
        self.bw_function = bw_function
        self.feature_selection_dict = {}
        
        # Private
        self._kde_list:list[KDE]
        self._kernel_density_dict:dict[KernelDensity]
        self._ranking_divergence:pd.DataFrame
        self._class_values:set
        self._bw:pd.DataFrame
        self._class_representation:dict


    def _bw_best_estimator(self, data:pd.DataFrame) -> float:
        range = abs(max(data) - min(data))
        len_unique = len(data.unique())
        params = {'bandwidth': np.linspace(range/len_unique, range, self.x_sample)}
        data = data.values[:, np.newaxis]
        grid = GridSearchCV(KernelDensity(), params, cv=3)
        grid.fit(data)
        return grid.best_estimator_.bandwidth


    def _calculate_bandwidth(self, X:pd.DataFrame, y:pd.Series) -> None:
        
        # Different types of function can be used
        if self.bw_function == self.BW_HSILVERMAN:
            bw_f = hselect.hsilverman
        elif self.bw_function == self.BW_HSCOTT:
            bw_f = hselect.hscott
        elif self.bw_function == self.BW_HSJ:
            bw_f = hselect.hsj
        else:
           bw_f = self._bw_best_estimator

        # Building the dataframe
        bw_list = []
        for v in X.columns:
            for c in self._class_values:
                bw_list.append([v, c, bw_f(X[y == c][v])])

        self._bw = pd.DataFrame(bw_list, columns=['variable','target','bandwidth'])


    def _background(f):
            def wrapped(*args, **kwargs):
                return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
            return wrapped


    @_background
    def _kde_lambda_asyncio(self, X:pd.DataFrame, y:pd.Series, c:str, v:str) -> None:
        data = X[y == c][v]
        minimum, maximum = data.min(), data.max()
        margin = (maximum-minimum) * self.margin_percentage
        x_points = np.linspace(minimum-margin, maximum+margin, self.x_sample)
        bw = self._bw[(self._bw.variable == data.name) & (self._bw.target == c)].bandwidth.values[0]
        kde = KernelDensity(kernel=self.kernel, bandwidth=bw).fit(data.values[:, np.newaxis])
        y_points = np.exp(kde.score_samples(x_points[:, np.newaxis]))
        self._kernel_density_dict[c][v] = kde
        self._kde_list.append(KDE(v, c, x_points, y_points))


    def _calculate_kde_asyncio(self, X:pd.DataFrame, y:pd.Series) -> None:
        self._kernel_density_dict, self._kde_list = {}, multiprocessing.Manager().list()
        cols = X.columns
        for c in self._class_values:
            self._kernel_density_dict[c] = {}
            for v in cols:
                self._kde_lambda_asyncio(X, y, c, v)


    def _normalize(self, data:list) -> list:
        s = sum(data)
        return list(map(lambda x: x/s if s!=0 else 0, data)) # range[0, 1] -> the sum is 1


    def _calculate_divergence(self) -> None:
        
        def hellinger_distance(p:list, q:list):
            return sqrt(1 - sum([sqrt(a*b) for a, b in zip(p, q)]))

        def kl_divergence(p:list, q:list) -> float:
            return sum([0 if (a == 0 or b < 2.2250738585072014e-308) else a * log2(a/b) for a,b in zip(p, q)])

        comb:list[tuple[KDE,KDE]]
        comb, scores = [], []
        while len(comb) == 0:
            comb = list(itertools.combinations(self._kde_list, 2))
        for kde1, kde2 in comb:
            t1, t2 = kde1.target, kde2.target
            f1, f2 = kde1.feature, kde2.feature
            if t1 != t2 and f1 == f2:
                p = self._normalize(kde1.y_points)
                q = self._normalize(kde2.y_points)
                y_avg =  np.mean( np.array([ p, q ]), axis=0)
                kl1 = kl_divergence(p, y_avg)
                kl2 = kl_divergence(q, y_avg)
                hellinger = hellinger_distance(p, q)
                scores.append([f1, t1, t2, (kl1+kl2)/2, hellinger])

        ranking = pd.DataFrame(scores, columns=['variable','p0','p1','indicator', 'hellinger']).drop_duplicates()
        score_total = ranking.indicator.sum()
        ranking['percentage'] = ranking.indicator / score_total
        ranking = ranking.sort_values(by='hellinger', ascending=False)
        self._ranking_divergence = ranking


    def _calculate_feature_selection(self):
        threshold = 1.0 - pow(10,-ceil(1+log10(len(self._ranking_divergence))))
        finished_class, dict_result = {}, {}
        for c in self._class_values:
            dict_result[c], finished_class[c] = {}, {}
            for c2 in self._class_values:
                if (c != c2):
                    finished_class[c][c2] = False
        def addDict(dict_result, class_1, class_2, variable, hellinger, finished_class):
            k = class_1+' || '+variable
            m = map(lambda x: x.split(' || ')[0], dict_result[class_2].keys())
            not_in_dict = class_1 not in set(m)
            if not finished_class[class_1][class_2]:
                if not_in_dict:
                    dict_result[class_2][k] = hellinger
                    finished_class[class_1][class_2] = hellinger >= threshold
                else:
                    class_list = list(m)
                    p = 1
                    for i in range(len(class_list)):
                        if (class_list[i] == class_1):
                            valor = list(dict_result[class_2].values())[i]
                            p *= (1-valor)
                    finished_class[class_1][class_2] = 1-p >= threshold
                    if not finished_class[class_1][class_2]:
                        dict_result[class_2][k] = hellinger
            return dict_result

        for _, row in self._ranking_divergence.iterrows():
            variable = row.variable
            class_1 = row.p0
            class_2 = row.p1
            hellinger = row.hellinger
            dict_result = addDict(dict_result, class_1, class_2, variable, hellinger, finished_class)
            dict_result = addDict(dict_result, class_2, class_1, variable, hellinger, finished_class)

        for d in dict_result:
            self.feature_selection_dict[d] = set(map(lambda x: x.split(' || ')[1],dict_result[d].keys()))


    def _calculate_target_representation(self, y:pd.Series) -> None:
        self._class_representation = {}
        for target in self._class_values:
            self._class_representation[target] = len([i for i in y if i==target]) / len(y)


    def fit(self, X:pd.DataFrame, y:pd.Series) -> None:
        self._class_values = set(y)
        self._calculate_target_representation(y)
        self._calculate_bandwidth(X, y)
        self._calculate_kde_asyncio(X, y)
        self._calculate_divergence()
        self._calculate_feature_selection()


    def predict(self, X:pd.DataFrame) -> list:
        mp.dps = 100

        # Iterating each test record
        y_pred = []
        for _, row in X.iterrows():
            # Calculating the probability of each class
            probabilities = [] # array with all the probabilities of each class
            y, m, s = None, 0, 0
            
            # Running through the final variables
            for c, variables in self.feature_selection_dict.items():
                kde_values = []
                for v in variables:
                    # We get the probabilities with KDE. Instead of x_sample (50) records, we pass this time only one
                    k = mp.exp(self._kernel_density_dict[c][v].score_samples(np.array([mp.mpf(str(row[v]))])[:, np.newaxis])[0])
                    kde_values.append(k)
                
                pr = prod(kde_values)
                probability = pr * self._class_representation[c] # The last operand is the number of times a record with that class is given in the train dataset
                probabilities.append((probability,c))
                s += probability
                # We save the class with a higher probability
                if probability > m:
                    m, y = probability, c
            
            if s > 0:
                p = list(map(lambda x: (x[0]/s, x[1]), probabilities))
                # y_pred.append((y, p))
                y_pred.append(y)
            else:
                # If none of the classes has a probability greater than zero, we assign the class that is most representative of the train dataset
                k = max(self._class_representation, key=self._class_representation.get)
                y_pred.append((k, self._class_representation[k]))
                print("Warn: inferred class. Row -> "+str(row).replace("\n"," "))

        return y_pred
            


            
    


    
