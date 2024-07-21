# Explainable Class–Specific Naive–Bayes Classifier

Explicable Naive Bayes (XNB) classifier includes two important
features: 

1) The probability is calculated by means of Kernel Density Estimation (KDE).

2) The probability for each class does not use all variables, but only those that are relevant for each specific class.

From the point of view of the classification performance, the XNB classifier is comparable to NB classifier.
However, the XNB classifier provides the subsets of relevant variables for each class, which contributes considerably to explaining how the predictive model is performing.
In addition, the subsets of variables generated for each class are usually different and with remarkably small cardinality.

## Example of use:

```python
from xnb import XNB
from xnb.enum import BWFunctionName, Kernel, Algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

''' 1. Read the dataset. It is important that the dataset is a pandas DataFrame object with named columns. This way, we can obtain the dictionary of important variables for each class.'''
df = pd.read_csv("path_to_data/iris.csv")
x, y = df.iloc[:, 0:-1], df.iloc[:,-1]
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

''' 2. By calling the fit() function, we prepare the object to be able to make the prediction later. '''
xnb = XNB()
xnb.fit(
  x_train,
  y_train,
  bw_function = BWFunctionName.BEST_ESTIMATOR, # optional
  kernel = Kernel.GAUSSIAN, # optional
  algorithm = Algorithm.AUTO, # optional
  n_sample = 50 # optional
)

''' 3. When the fit() function finishes, we can now access the feature selection dictionary it has calculated. '''
feature_selection = xnb.feature_selection_dict

''' 4. We predict the values of "y_test" using implicitly the calculated dictionary. '''
y_pred = xnb.predict(X_test)

# Output
print(feature_selection)
print(accuracy_score(y_test, y_pred))
```
