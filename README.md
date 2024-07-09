# KDE Classifier

## 1. Example of use:

```python
from kde_classifier.stratified_naive_bayes import XNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

''' 1. Read the dataset. '''
df = pd.read_csv("data/iris.csv")
X, y = df.iloc[:, 0:-1], df.iloc[:,-1]
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

''' 2. By calling the fit() function, we prepare the class to be able to make the prediction later. '''
xnb = XNB()
xnb.fit(x_train, y_train)

''' 3. When the fit() function finishes, we can now access the feature selection it has calculated. '''
feature_selection = xnb.feature_selection_dict

''' 4. We predict the values of "y_test" thanks to the calculated dictionary. '''
y_pred = xnb.predict(X_test)

# Output
print(feature_selection)
print(accuracy_score(y_test, y_pred))
```

 ## 2. XNB documentation

 ### 2.1 Parameters
| Name   | Default Value | Description |
| ----------- | ----------- | ----------- |
| kernel | gaussian | Type of Kernel for the KDE analysis. You could use the XNB.K_GAUSSIAN constant. |
| margin_percentage | 0 | Margin to the left and right of the series of values in a column. In case you are interested in applying KDE only to a proportion of the data. |
| x_sample | 50 | Number of equidistant values between the minimum and maximum of a column. These will be used for the KDE calculation. |
| bw_function | hsilverman | Bandwidth function. You could use: XNB.BW_HSILVERMAN, XNB.BW_HSCOTT, XNB.BW_HSJ or XNB.BW_BEST_ESTIMATOR constants. |


### 2.2 Functions

```
xnb.fit(x_train, y_train)
```
This function prepares the XNB class object to generate the dictionary with the best variables to predict each class.

<b>Parameters:</b>

| Name   | Type | Description |
| ----------- | ----------- | ----------- |
| x_train | Pandas.DataFrame | Training data, only feature columns |
| y_train | Pandas.Series | Training target values |

<b>Returns:</b> None

```
xnb.predict(X_test)
```
This function uses the previously calculated dictionary (feature selection) to return a prediction of the y_test.

<b>Parameters:</b>

| Name   | Type | Description |
| ----------- | ----------- | ----------- |
| X_test | Pandas.DataFrame | Testing data, only feature columns |

<b>Returns:</b>
| Name   | Type | Description |
| ----------- | ----------- | ----------- |
| y_pred | Pandas.Series | Predicted target values |

### 2.3 Attributes
| Name   | Description |
| ----------- | ----------- |
| kernel  | kernel used |
| margin_percentage | margin used |
| x_sample | x_sample used |
| bw_function | bandwidth function used |
