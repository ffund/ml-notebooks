---
title: 'Hyperparameter optimization'
author: 'Fraida Fund'
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Hyperparameter optimization

*Fraida Fund*
:::


::: {.cell .code }
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import scipy

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform
```
:::


::: {.cell .markdown }

## Grid search

:::


::: {.cell .markdown }
For models with a single hyperparameter controlling bias-variance (for
example: $k$ in $k$ nearest neighbors), we used sklearns\'s `KFoldCV` or
`validation_curve` to test a range of values for the hyperparameter, and
to select the best one.

When we have *multiple* hyperparameters to tune, we can use
`GridSearchCV` to select the best *combination* of them.

For example, we saw three ways to tune the bias-variance of an
SVM classifier:

-   Changing the kernel
-   Changing $C$
-   For an RBF kernel, changing $\gamma$

To get the best performance from an SVM classifier, we need to find the
best *combination* of these hyperparameters. This notebook shows how to
use `GridSearchCV` to tune an SVM classifier.
:::

::: {.cell .markdown }
We will work with a subset of the MNIST handwritten digits data. First,
we will get the data, and assign a small subset of samples to training
and test sets.
:::

::: {.cell .code }
```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True )
```
:::

::: {.cell .code }
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=300)
```
:::

::: {.cell .markdown }
Let's try this initial parameter "grid":
:::

::: {.cell .code }
```python
param_grid = [
  {'C': [0.1, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1000], 'gamma': [0.01, 0.0001], 'kernel': ['rbf']},
 ]
param_grid
```
:::

::: {.cell .markdown }
Now we'll set up the grid search. We can use `fit` on it, just like any
other `sklearn` model.

I added `return_train_score=True` to my `GridSearchSV` so that it will
show me training scores as well:
:::

::: {.cell .code }
```python
clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True)
%time clf.fit(X_train, y_train)
```
:::

::: {.cell .markdown }
Here are the results:
:::

::: {.cell .code }
```python
pd.DataFrame(clf.cv_results_)
```
:::

::: {.cell .markdown }
To inform our search, we will use our understanding of how SVMs work,
and especially how the $C$ and $\gamma$ parameters control the bias and
variance of the SVM.
:::

::: {.cell .markdown }
### Linear kernel
:::

::: {.cell .markdown }
Let's tackle the linear SVM first, since it's faster to fit. We
didn't see any change in the accuracy when we vary $C$. So, we should
extend the range of $C$ over which we search.

I'll try higher and lower values of $C$, to see what happens.
:::

::: {.cell .code }
```python
param_grid = [
  {'C': [1e-6, 1e-4, 1e-2, 1e2, 1e4, 1e6], 'kernel': ['linear']},
 ]
param_grid
```
:::

::: {.cell .code }
```python
clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True)
%time clf.fit(X_train, y_train)
```
:::

::: {.cell .code }
```python
pd.DataFrame(clf.cv_results_)
```
:::

::: {.cell .code }
```python
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_C', y='mean_train_score', label="Training score");
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_C', y='mean_test_score', label="Validation score");
plt.xscale('log');
```
:::

::: {.cell .markdown }
It looks like we get a slightly better validation score near the smaller
values for $C$! What does this mean?

Let's try:
:::

::: {.cell .code }
```python
param_grid = [
  {'C': np.linspace(1e-5, 1e-7, num=10), 'kernel': ['linear']},
 ]
param_grid
```
:::

::: {.cell .code }
```python
clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True)
%time clf.fit(X_train, y_train)
```
:::

::: {.cell .code }
```python
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_C', y='mean_train_score', label="Training score");
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_C', y='mean_test_score', label="Validation score");
plt.xscale('log');
```
:::

::: {.cell .markdown }
We can be satisfied that we have found a good hyperparameter only when
we see the high bias AND high variance side of the validation curve!
:::

::: {.cell .markdown }
### RBF kernel
:::

::: {.cell .markdown }
Now, let's look at the RBF kernel.

In our first search, the accuracy of the RBF kernel is very poor. We may
have high bias, high variance, (or both).
:::

::: {.cell .markdown }
When $C=0.1$ in our first search, both training and validation scores
were low. This suggests high bias.

When $C=1000$ in our first search, training scores were high and
validation scores were low. This suggests high variance.
:::

::: {.cell .markdown }
What next? We know from our discussion of bias and variance of SVMs that
to combat overfitting, we can decrease $\gamma$ and/or decrease $C$.

For now, let's keep the higher value of $C$, and try to reduce the
overfitting by decreasing $\gamma$.
:::

::: {.cell .code }
```python
param_grid = [
  {'C': [1000], 'gamma': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11], 'kernel': ['rbf']},
 ]
param_grid
```
:::

::: {.cell .code }
```python
clf = GridSearchCV(SVC(), param_grid, cv=2, refit=True, verbose=100, n_jobs=-1, return_train_score=True)
%time clf.fit(X_train, y_train)
```
:::

::: {.cell .code }
```python
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_gamma', y='mean_train_score', label="Training score")
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x='param_gamma', y='mean_test_score', label="Validation score")
plt.xscale('log');
```
:::

::: {.cell .markdown }
Here, we see that (at least for $C=1000$), values of $\gamma$ greater
than `1e-5` seem to overfit, while decreasing $\gamma$ lower than
`1e-10` may underfit.

But we know that changing $C$ also affects the bias variance tradeoff!
For different values of $C$, the best value of $\gamma$ will be
different, and there may be a better *combination* of $C$ and $\gamma$
than any we have seen so far. We can try to increase and decrease $C$ to
see if that improves the validation score.
:::

::: {.cell .markdown }
Now that we have a better idea of where to search, we can set up our
"final" search grid.

We know that to find the best validation accuracy for the linear kernel,
we should make sure our search space includes `1e-6` and `1e-7`. I chose
to vary $C$ from `1e-8` to `1e-4`. (I want to make sure the best value
is not at the edge of the search space, so that we can be sure there
isn't a better value if we go lower/higher.)

We know that to find the best validation accuracy for the RBF kernel, we
should make sure our search space includes $\gamma$ values around `1e-6`
and `1e-7` when $C=1000$. For larger values of $C$, we expect that
we'll get better results with smaller values of $\gamma$. For smaller
values of $C$, we expect that we'll get better results with larger
values of $\gamma$. I chose to vary $C$ from `1` to `1e6` and $\gamma$
from `1e-4` to `1e-11`.

That's a big search grid, so this takes a long time to fit! (Try this
at home with a larger training set to get an idea\...)
:::

::: {.cell .code }
```python
param_grid = [
  {'C': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4], 'kernel': ['linear']},
  {'C': [1, 1e2, 1e3, 1e4, 1e5, 1e6], 'gamma': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11], 'kernel': ['rbf']},
 ]
param_grid
```
:::

::: {.cell .code }
```python
clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True)
%time clf.fit(X_train, y_train)
```
:::

::: {.cell .markdown }
For the linear kernel, here\'s what we found:
:::

::: {.cell .code }
```python
df_cv   = pd.DataFrame(clf.cv_results_)
df_cv = df_cv[df_cv['param_kernel']=='linear']
```
:::

::: {.cell .code }
```python
sns.lineplot(data=df_cv, x='param_C', y='mean_train_score', label="Training score")
sns.lineplot(data=df_cv, x='param_C', y='mean_test_score', label="Validation score")
plt.xscale('log');
```
:::

::: {.cell .markdown }
For the RBF kernel, here\'s what we found:
:::

::: {.cell .code }
```python
df_cv   = pd.DataFrame(clf.cv_results_)
df_cv = df_cv[df_cv['param_kernel']=='rbf']

plt.figure(figsize=(12,5))

ax1=plt.subplot(1,2,1)
pvt = pd.pivot_table(df_cv, values='mean_test_score', index='param_C', columns='param_gamma')
sns.heatmap(pvt, annot=True, cbar=False, vmin=0, vmax=1, cmap='PiYG');
plt.title("Validation scores");

ax2=plt.subplot(1,2,2, sharey=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)
pvt = pd.pivot_table(df_cv, values='mean_train_score', index='param_C', columns='param_gamma')
sns.heatmap(pvt, annot=True, cbar=False, vmin=0, vmax=1, cmap='PiYG');
plt.title("Training scores");
```
:::

::: {.cell .markdown }
We see that $\gamma$ and $C$ control the bias-variance tradeoff of the
SVM model as follows.

-   In the top left region, $C$ is small (the margin is wider) and
    $\gamma$ is small (the kernel bandwidth is large). In this region,
    the model has more bias (is prone to underfit). The validation
    scores and training scores are both low.
-   On the right side (and we\'d expect to see this on the bottom right
    if we extend the range of $C$ even higher), $C$ is large (the margin
    is narrower) and $\gamma$ is large (the kernel bandwidth is small.
    In this region, the model has more variance (is likely to overfit).
    The validation scores are low, but the training scores are high.

In the middle, we have a region of good combinations of $C$ and
$\gamma$.

Since the parameter grid above shows us the validation accuracy
decreasing both as we increase each parameter\* and also as we decrease
each parameter, we can be a bit more confident that we captured the
point in the bias-variance surface where the error is smallest.

\* $C$ is different because increasing $C$ even more may not actually
change the margin.
:::

::: {.cell .markdown }
We can see the "best" parameters, with which the model was re-fitted:
:::

::: {.cell .code }
```python
print(clf.best_params_)
```
:::

::: {.cell .markdown }
And we can evaluate the re-fitted model on the test set. (Note that the
`GridSearchCV` only used the training set; we have not used the test set
at all for model fitting.)
:::

::: {.cell .code }
```python
y_pred = clf.predict(X_test)
```
:::

::: {.cell .code }
```python
accuracy_score(y_pred, y_test)
```
:::

::: {.cell .markdown }

## Random search

:::


::: {.cell .markdown }

Our grid search found a pretty good set of hyperparameters, but it took a long time - about 100 seconds.

With a random search, we may be able to find hyperparameters that are still pretty good, in much less time.

:::


::: {.cell .markdown}
We will search a similar range of parameters, although focusing only on the RBF kernel.  But instead of specifying points on a grid like 

```
param_grid = [
  {'C': [1, 1e2, 1e3, 1e4, 1e5, 1e6], 'gamma': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11], 'kernel': ['rbf']},
 ]
```

we will specify distributions from which to sample:
:::


::: {.cell .code}
```python
param_grid = [
  {'C': loguniform(1, 1e6), 'gamma': loguniform(1e-11, 1e-4), 'kernel': ['rbf']},
 ]
```
:::


::: {.cell .markdown}

and then we will specify the total number of points to sample - 10, in this example:
:::

::: {.cell .code}
```python
clf_rnd = RandomizedSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True,  n_iter = 10)
%time clf_rnd.fit(X_train, y_train)
```
:::


::: {.cell .code}
```python
pd.DataFrame(clf_rnd.cv_results_)
```
:::


::: {.cell .code}
```python
print(clf_rnd.best_params_)
```
:::


::: {.cell .code}
```python
y_pred = clf_rnd.predict(X_test)
accuracy_score(y_pred, y_test)
```
:::

::: {.cell .markdown}

To see how this works, we will re-run the random search with more iterations than we really need, just so that we can visualize how it searches the hyperparameter space.
:::

::: {.cell .code}
```python
clf_rnd = RandomizedSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True,  n_iter = 50)
%time clf_rnd.fit(X_train, y_train)
```
:::

::: {.cell .code}
```python
pvt = pd.pivot_table(df_cv, values="mean_test_score",
                     index="param_C", columns="param_gamma")

ax = sns.heatmap(pvt, annot=False, cbar=False, cmap="PiYG", vmin=0, vmax=1)
ax.set_title("Validation scores + RandomSearch")

df_rnd = pd.DataFrame(clf_rnd.cv_results_)
df_rnd = df_rnd[df_rnd["param_kernel"] == "rbf"]

ax2 = ax.twiny().twinx()
ax2.set_xscale("log");  ax2.set_yscale("log")

# hardcoded padded limits
ax2.set_xlim(3.1623e-12, 3.1623e-04)
ax2.set_ylim(3.1623e+06, 3.1623e-01)

for i, (g, c) in enumerate(zip(df_rnd["param_gamma"], df_rnd["param_C"])):
    ax2.scatter(g, c, s=200, facecolors="none", edgecolors="black")
    ax2.text(g, c, i, ha="center", va="center", fontsize=8)
```
:::

::: {.cell .markdown}

Our random search can find a good solution, in only about ~20 seconds. However, depending on the random samples it chooses, it may be a better solution or a worse solution than the one we found via grid search.

:::



::: {.cell .markdown }

## Adaptive Search (Bayes Search)

:::


::: {.cell .markdown }

Finally, we'll consider one other type of hyperparameter optimization: we will look at an adaptive search that uses information about the models it has seen so far in order to decide which part of the hyperparameter space to sample from next.

We will install the `scikit-optimize` package, which provides `BayesSearchCV`.

:::



::: {.cell .code}
```python
!pip install scikit-optimize
```
:::


::: {.cell .code}
```python
from skopt import BayesSearchCV
```
:::


::: {.cell .markdown}
We will define the search space:
:::


::: {.cell .code}
```python
param_grid = [
  {'C': (1, 1e6, 'log-uniform'), 'gamma': (1e-11, 1e-4, 'log-uniform'), 'kernel': ['rbf']},
 ]
```
:::


::: {.cell .markdown}

As before, we will specify the total number of points to sample - 5, in this example:

:::

::: {.cell .code}
```python
clf_bys = BayesSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=100, n_jobs=-1, return_train_score=True,  n_iter = 5)
%time clf_bys.fit(X_train, y_train)
```
:::


::: {.cell .code}
```python
pd.DataFrame(clf_bys.cv_results_)
```
:::


::: {.cell .code}
```python
print(clf_bys.best_params_)
```
:::


::: {.cell .code}
```python
y_pred = clf_bys.predict(X_test)
accuracy_score(y_pred, y_test)
```
:::


::: {.cell .markdown}

To see how this works, we will re-run the Bayes search with more iterations than we really need, just so that we can visualize how it searches the hyperparameter space.
:::



::: {.cell .code}
```python
clf_bys = BayesSearchCV(SVC(), param_grid, cv=3, refit=False, verbose=100, n_jobs=-1, n_iter = 50)
clf_bys.fit(X_train, y_train)
```
:::


::: {.cell .code}
```python
pd.DataFrame(clf_bys.cv_results_)
```
:::

::: {.cell .code}
```python
pvt = pd.pivot_table(df_cv, values="mean_test_score",
                     index="param_C", columns="param_gamma")

ax = sns.heatmap(pvt, annot=False, cbar=False, cmap="PiYG", vmin=0, vmax=1)
ax.set_title("Validation scores + BayesSearch")

df_bys = pd.DataFrame(clf_bys.cv_results_)
df_bys = df_bys[df_bys["param_kernel"] == "rbf"]

ax2 = ax.twiny().twinx()
ax2.set_xscale("log");  ax2.set_yscale("log")

# hardcoded half-decade padded limits
ax2.set_xlim(3.1623e-12, 3.1623e-04)
ax2.set_ylim(3.1623e+06, 3.1623e-01)

# scatter + labels
for i, (g, c) in enumerate(zip(df_bys["param_gamma"], df_rnd["param_C"])):
    ax2.scatter(g, c, s=200, facecolors="none", edgecolors="black")
    ax2.text(g, c, i, ha="center", va="center", fontsize=8)
```
:::
