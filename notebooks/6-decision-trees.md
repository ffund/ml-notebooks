---
title: 'Demo: Decision trees'
author: 'Fraida Fund'
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Demo: Decision trees

*Fraida Fund*
:::

::: {.cell .markdown }
This is a simple demo notebook that demonstrates a decision tree
classifier.

**Attribution**: Parts of this notebook are slightly modified from [this
tutorial from "Intro to Data
Mining"](http://www.cse.msu.edu/~ptan/dmbook/tutorials/tutorial6/tutorial6.html).
:::

::: {.cell .code }
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
```
:::

::: {.cell .code }
```python
df = pd.read_csv('http://www.cse.msu.edu/~ptan/dmbook/tutorials/tutorial6/vertebrate.csv')
df
```
:::

::: {.cell .markdown }
We'l make it a binary classification problem:
:::

::: {.cell .code }
```python
df['Class'] = df['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
df
```
:::

::: {.cell .markdown }
## Decision tree
:::

::: {.cell .code }
```python
y = df['Class']
X = df.drop(['Name','Class'],axis=1)

clf_dt = DecisionTreeClassifier(criterion='entropy')
clf_dt = clf_dt.fit(X, y)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(10,10))
sklearn.tree.plot_tree(clf_dt, 
                    feature_names = df.columns.drop(['Name', 'Class']),
                    class_names = ["mammals", "non-mammals"],
                    filled=True, rounded=True);
```
:::

::: {.cell .markdown }
### Feature importance
:::

::: {.cell .code }
```python
df_importance = pd.DataFrame({'feature': df.columns.drop(['Name', 'Class']),
                              'importance': clf_dt.feature_importances_})
df_importance
```
:::
