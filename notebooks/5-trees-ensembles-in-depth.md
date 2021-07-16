---
title: 'Demo: Decision trees and ensembles'
author: 'Fraida Fund'
jupyter:
  colab:
    name: '5-trees-ensembles-in-depth.ipynb'
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
# Demo: Decision trees and ensembles

*Fraida Fund*
:::

::: {.cell .markdown }
This is a simple demo notebook that demonstrates a decision tree
classifier or an ensemble of decision trees.

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

::: {.cell .markdown }
## Bagged tree
:::

::: {.cell .code }
```python
n_tree = 3
clf_bag = BaggingClassifier(n_estimators=n_tree)
clf_bag = clf_bag.fit(X, y)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(n_tree*8, 10))
for idx, clf_t in enumerate(clf_bag.estimators_):
  plt.subplot(1, n_tree,idx+1)
  sklearn.tree.plot_tree(clf_t, 
                      feature_names = df.columns.drop(['Name', 'Class']),
                      class_names = ["mammals", "non-mammals"],
                      filled=True, rounded=True)  
```
:::

::: {.cell .markdown }
Notice the similarities! The bagged trees are highly correlated.
:::

::: {.cell .markdown }
Let's look at the bootstrap sets each tree was trained on:
:::

::: {.cell .code }
```python
for samples in clf_bag.estimators_samples_:
  print(df.iloc[samples])
```
:::

::: {.cell .markdown }
## Random forest
:::

::: {.cell .code }
```python
n_tree = 3
clf_rf = RandomForestClassifier(n_estimators=n_tree, )
clf_rf = clf_rf.fit(X, y)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(n_tree*8, 10))
for idx, clf_t in enumerate(clf_rf.estimators_):
  plt.subplot(1, n_tree,idx+1)
  sklearn.tree.plot_tree(clf_t, 
                      feature_names = df.columns.drop(['Name', 'Class']),
                      class_names = ["mammals", "non-mammals"],
                      filled=True, rounded=True)  
```
:::

::: {.cell .markdown }
These trees are much less correlated.
:::

::: {.cell .markdown }
## AdaBoost
:::

::: {.cell .code }
```python
n_tree = 3
clf_ab = AdaBoostClassifier(n_estimators=n_tree)
clf_ab = clf_ab.fit(X, y)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(n_tree*8, 10))
for idx, clf_t in enumerate(clf_ab.estimators_):
  plt.subplot(1, n_tree,idx+1)
  sklearn.tree.plot_tree(clf_t, 
                      feature_names = df.columns.drop(['Name', 'Class']),
                      class_names = ["mammals", "non-mammals"],
                      filled=True, rounded=True)  
```
:::

::: {.cell .markdown }
The output will be a weighted average of the predictions of all three
trees.
:::

::: {.cell .markdown }
As we add more trees, the ensemble accuracy increases:
:::

::: {.cell .code }
```python
for p in clf_ab.staged_predict(X):
  print(np.mean(p==y))
```
:::
