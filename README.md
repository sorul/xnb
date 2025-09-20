# Explainable Class–Specific Naive–Bayes Classifier

![Test](https://github.com/sorul/xnb/actions/workflows/testing_coverage.yml/badge.svg?branch=master)
![codecov.io](https://codecov.io/github/sorul/xnb/coverage.svg?branch=master)


## Description
Explainable Naive Bayes (XNB) classifier includes two important
features: 

1) The probability is calculated by means of Kernel Density Estimation (KDE).

2) The probability for each class does not use all variables,
but **only those that are relevant** for each specific class.

From the point of view of the classification performance,
the XNB classifier is comparable to NB classifier.
However, the XNB classifier provides the subsets of relevant variables for each class,
which contributes considerably to explaining how the predictive model is performing.
In addition, the subsets of variables generated for each class are usually different and with remarkably small cardinality.

## Installation

For example, if you are using pip, yo can install the package by:
```
pip install xnb
```

## Example of use:

```python
from xnb import XNB
from xnb.enums import BWFunctionName, Kernel, Algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
''' 1. Read the dataset.
It is important that the dataset is a pandas DataFrame object with named columns.
This way, we can obtain the dictionary of important variables for each class.'''
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
x = df.drop('target', axis=1)
y = df['target'].replace(to_replace=[0, 1, 2],
                         value=['setosa', 'versicolor', 'virginica'])
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=0,
)
''' 2. By calling the fit() function,
we prepare the object to be able to make the prediction later. '''
# Initialize and fit the XNB model
xnb = XNB(
    show_progress_bar=False,
    bw_function=BWFunctionName.HSILVERMAN,
    kernel=Kernel.GAUSSIAN,
    algorithm=Algorithm.AUTO,
    n_sample=50,
)

# Fit the model
xnb.fit(x_train, y_train)
''' 3. When the fit() function finishes,
we can now access the feature selection dictionary it has calculated. '''
feature_selection = xnb.feature_selection_dict
''' 4. We predict the values of "y_test" using implicitly the calculated dictionary. '''
y_pred = xnb.predict(x_test)

# Output
print('Relevant features for each class:\n')
for target, features in feature_selection.items():
  print(f'{target}: {features}')
print(f'\n-------------\nAccuracy: {accuracy_score(y_test, y_pred)}')
```
The output is:
```
Relevant features for each class:

setosa: {'petal length (cm)'}
virginica: {'petal length (cm)', 'petal width (cm)'}
versicolor: {'petal length (cm)', 'petal width (cm)'}

-------------
Accuracy: 1.0
```

# Links
[![GitHub](https://img.shields.io/badge/GitHub-Repository-negro?style=for-the-badge&logo=github)](https://github.com/sorul/xnb)
[![PyPI](https://img.shields.io/badge/PyPI-Package-3776AB?style=for-the-badge&logo=pypi)](https://pypi.org/project/xnb/)