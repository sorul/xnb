import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool, Manager
import mpmath as mp
import itertools
from math import log2, prod, ceil, log10, sqrt
from kde_classifier import _bandwidth_functions as bf
from kde_classifier._kde_object import KDE
import copy


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
        fature_selection_dict = {}
        self.feature_selection_dict = fature_selection_dict
        
        # Private
        self._X:pd.DataFrame
        self._y:pd.Series
        self._kde_list:list[KDE]
        self._ranking_divergence:pd.DataFrame
        self._class_values:set
        self._bw:pd.DataFrame
        self._class_representation:dict
        self._bw_list:list

    
    def _calculate_bandwidth(self) -> None:
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
            raise ValueError("'"+self.bw_function+"' is not a valid value for a bandwidth function.")

        # Building the dataframe
        self._bw_list = []
        for v in self._X.columns:
            for c in self._class_values:
                self._bw_list.append([v, c, bw_f(self._X[self._y == c][v], self.x_sample)])

        self._bw = pd.DataFrame(list(self._bw_list), columns=['variable', 'target', 'bandwidth'])

    
    def _kde_lambda(self, c:str, v:str) -> KDE:
        feature_data = self._X[v]
        minimum, maximum = feature_data.min(), feature_data.max()
        data = self._X[self._y == c][v]
        margin = (maximum-minimum) * self.margin_percentage
        x_points = np.linspace(minimum-margin, maximum+margin, self.x_sample)
        bw = self._bw[(self._bw.variable == data.name) & (self._bw.target == c)].bandwidth.values[0]
        kde = KernelDensity(kernel=self.kernel, bandwidth=bw).fit(data.values[:, np.newaxis])
        y_points = np.exp(kde.score_samples(x_points[:, np.newaxis]))
        return KDE(v, c, kde, x_points, y_points)

    
    def _calculate_kde(self, comb:list[tuple[KDE,KDE]] = None) -> None:
        comb = list(itertools.product(self._class_values, self._X.columns)) if comb is None else comb
        with Pool() as p:
            self._kde_list = p.starmap(self._kde_lambda, comb)
            p.close()

    
    def _normalize(self, data:list) -> list:
        s = sum(data)
        return list(map(lambda x: x/s if s!=0 else 0, data)) # range[0, 1] -> the sum is 1

    
    def _calculate_divergence(self) -> None:
        def hellinger_distance(p:list, q:list):
            s = sum([sqrt(a*b) for a, b in zip(p, q)])
            assert s >= 0 and s<=1, f"Bhattacharyya coefficient in [0,1] expected, got: {s}"
            return sqrt(1-s)
        kde_dict = {}
        for v in self._X.columns:
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
        ranking = pd.DataFrame(scores, columns=['variable','p0','p1', 'hellinger']).drop_duplicates()
        ranking = ranking.sort_values(by=['hellinger','variable','p0','p1'], ascending=False)
        self._ranking_divergence = ranking

    
    def _calculate_feature_selection(self):
        self.feature_selection_dict = {}
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
            not_in_dict = class_1 not in set(copy.deepcopy(m))
            if not finished_class[class_1][class_2]:
                if not_in_dict:
                    dict_result[class_2][k] = hellinger
                    finished_class[class_1][class_2] = hellinger >= threshold
                else:
                    class_list = list(copy.deepcopy(m))
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
        self._X = X
        self._y = y
        self._class_values = set(y)
        self._calculate_target_representation(y)
        self._calculate_bandwidth()
        self._calculate_kde()
        self._calculate_divergence()
        self._calculate_feature_selection()


    def predict(self, X:pd.DataFrame) -> list:
        
        if len(self.feature_selection_dict) == 0:
            raise NotFittedError("This Stratified Naive Bayes instance is not fitted yet. Call 'fit' with appropiate arguments before using this estimator.")

        mp.dps = 100
        # Calculating KDE values only with the selected features in fitting process
        fsd = self.feature_selection_dict
        comb = [(k, v) for k in fsd for v in fsd[k]]
        self._calculate_kde(comb=comb)
        kde_dict = {}
        new_cols = list({x for v in fsd.values() for x in v})
        for v in new_cols:
            kde_dict[v] = {}
        for kde in self._kde_list:
            kde_dict[kde.feature][kde.target] = kde.kernel_density

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
                    k = mp.exp(kde_dict[v][c].score_samples(np.array([mp.mpf(str(row[v]))])[:, np.newaxis])[0])
                    kde_values.append(k)
                
                pr = prod(kde_values)
                probability = pr * self._class_representation[c] # The last operand is the number of times a record with that class is given in the train dataset
                probabilities.append((probability,c))
                s += probability
                # We save the class with a higher probability
                if probability > m:
                    m, y = probability, c
            
            if s > 0:
                p = list(map(lambda x: (x[0]/s, x[1]), probabilities)) # The probability of being the class
                # y_pred.append((y, p))
                y_pred.append(y)
            else:
                # If none of the classes has a probability greater than zero, we assign the class that is most representative of the train dataset
                k = max(self._class_representation, key=self._class_representation.get)
                y_pred.append((k, self._class_representation[k]))
                print("Warn: inferred class. Row -> "+str(row).replace("\n"," "))

        return y_pred
            

class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """