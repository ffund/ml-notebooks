---
title:  'Model selection in depth'
author: 'Fraida Fund'
jupyter:
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 4
---


::: {.cell .markdown}

# Model selection

We know that (in the "classic" view), the test error of a model tends
to decrease and then increase as we increase model complexity. For low
complexity, the bias dominates; for high complexity, the variance
dominates.

The training error, however, only decreases with increasing model
complexity. If we use training error to select a model, we'll select a
model that overfits. And during training, when we select a model, only
the training error is available to us.

![Image from "Elements of Statistical
Learning"](https://i.stack.imgur.com/alkeM.png)

*Image source: Elements of Statistical Learning*
:::

::: {.cell .markdown}
The solution is cross validation. Until now, we have been dividing our
data into two parts:

-   Training data: used to train the model
-   Test data: used to evaluate the performance of our model on new,
    unseen data

Now, we will make one more split:

-   Training data: used to train the model
-   Validation data: used to select the model complexity (usually by
    tuning some *hyperparameters*)
-   Test data: used to evaluate the performance of our model on new,
    unseen data
:::

::: {.cell .markdown}
Furthermore, we will refine this idea in order to reduce the dependence
on the particular samples we choose for the training, and to increase
the number of samples available for training. In K-fold cross
validation, we split the data into $K$ parts, each part being
approximately equal in size. For each split, we fit the data on $K-1$
parts and test the data on the remaining part. Then, we average the
score over the $K$ parts.
:::

::: {.cell .markdown}

For example, for $K=5$, it might look like this:

![](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

:::


::: {.cell .code}
```python
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from tqdm import tqdm
from ipywidgets import interact, fixed, widgets
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
:::



::: {.cell .markdown}
### Model selection using best K-Fold CV score

First, we will try to use K-fold CV to select a polynomial model to fit
the data in our first example.

We will use the `scikit-learn` module for K-fold CV.
:::

::: {.cell .code}
```python
def generate_polynomial_regression_data(n=100, xrange=[-1,1], coefs=[1,0.5,0,2], sigma=0.5):
  x = np.random.uniform(xrange[0], xrange[1], n)
  y = np.polynomial.polynomial.polyval(x,coefs) + sigma * np.random.randn(n)

  return x.reshape(-1,1), y
```
:::

::: {.cell .code}
```python
coefs=[1, 0.5, 0, 2.5]
n_samples = 100
sigma = 0.4

# generate polynimal data
x, y = generate_polynomial_regression_data(n=n_samples, coefs=coefs, sigma=sigma)

# divide into training and test set
# (we will later divide training data again, into training and validation set)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


@interact(d = widgets.IntSlider(min=1, max=25, value=1), show_train = True, show_test = False)
def plot_poly_fit(d, show_train, show_test,
                  xtr = fixed(x_train), ytr = fixed(y_train), 
                  xts = fixed(x_test), yts = fixed(y_test)):
  
  xtr_trans = np.power(xtr, np.arange(0, d))

  if show_train:
    sns.scatterplot(x = xtr.squeeze(), y = ytr);
  if show_test:
    sns.scatterplot(x = xts.squeeze(), y = yts);
  reg = LinearRegression().fit(xtr_trans, ytr)
  ytr_hat = reg.predict(xtr_trans)

  mse_tr = metrics.mean_squared_error(ytr, ytr_hat)
  mse_ts = metrics.mean_squared_error(yts, reg.predict(np.power(xts, np.arange(0, d))))

  sns.lineplot(x = xtr.squeeze(), y = ytr_hat, color='red')
  plt.xlabel('x');
  plt.ylabel('y');
  plt.title("Training MSE: %f\nTest MSE: %f" % (mse_tr, mse_ts))
```
:::


::: {.cell .markdown}
### Cross validation
:::


::: {.cell .markdown}

In this section, we will explore the use of K-fold cross validation for model selection

K-fold cross validation is sometimes used for model evaluation, and sometimes used for model selection. There are a few important differences in how K-fold CV is used in each case:

* **Model evaluation**: When the total number of samples available is very small, we may not want to split off a single held-out test set for evaluation, since the results will vary dramatically depending on the draw of training vs. test samples. Under these circumstances, we may prefer to use K-fold CV to evaluate the model. In this case, we pass the entire dataset to the K-fold CV.
* **Model selection**: When we want to select the best model out of a set of possible candidate models (or equivalently, select model hyperparamaters, such as degree of a polynomial model or number of knots in a spline model), we need a validation set to help us evaluate each candidate model on data not used to fit the model parameters. In this case, we _only_ pass the training subset of the data to the K-fold CV. We won't use the test set at all inside the K-fold cross validation, because using the test set for model selection is a form of data leakage.


:::

::: {.cell .markdown}

Splitting a data set for K-fold cross validation is conceptually very simple. The basic idea is:

* We get a list of indices of training data, and decide how many "folds" we will use. The number of validation samples in each fold $N_{val}$ wil be the total number of training samples, divided by the number of folds.
* Then, we iterate over the number of folds. In the first fold, we put the first $N_{val}$ samples in the validation set and exclude them from the training set. In the second fold, we put the second batch of $N_{val}$ samples in the validation set, and exclude them from the training set. Continue until $K$ folds.


In most circumstances, we will shuffle the list of training data indices first.

The `scikit-learn` library provides a `KFold` that does this for us:

:::


::: {.cell .code}
```python
nfold = 5
kf = KFold(n_splits=nfold,shuffle=True)

for isplit, idx in enumerate(kf.split(x_train)):     
    idx_tr, idx_val = idx 
```
:::

::: {.cell .markdown}

although it's also easy to do this ourselves:

:::

::: {.cell .code}
```python
nfold = 5                                   # number of folds (you choose!)
nval = x_train.shape[0]//nfold              # number of validation samples per fold
idx_split = [i*nval for i in range(nfold)]  
idx_list = np.arange(x_train.shape[0])      # list of training data indices
np.random.shuffle(idx_list)                 # shuffle list of indices

for i, idx in enumerate(idx_split):
  idx_val = idx_list[idx:idx+nval]
  idx_tr = np.delete(idx_list, idx_val)
```
:::

::: {.cell .markdown}

The outer loop can be used to divide the data into training and validation, but then we'll also need 
an inner loop to train and validate each model for this particular fold. 

In this case, suppose we want to evaluate polynomial models with different model orders from 

$$d=1, \quad \hat{y} = w_0 + w_1 x$$

to 

$$d=10, \quad \hat{y} = w_0 + w_1 x + w_2 x^2 + \ldots + w_{10} x^{10}$$

We could do something like this:

:::

::: {.cell .code}
```python
# create a k-fold object
nfold = 5
kf = KFold(n_splits=nfold,shuffle=True)

# model orders to be tested
dmax = 10
dtest_list = np.arange(1,dmax+1)
nd = len(dtest_list)

for isplit, idx in enumerate(kf.split(x_train)):     
  idx_tr, idx_val = idx 

  for dtest in dtest_list:
    # get "transformed" training and validation data
    x_train_dtest =  x_train[idx_tr]**np.arange(1,dtest+1)
    y_train_kfold =  y_train[idx_tr]
    x_val_dtest   =  x_train[idx_val]**np.arange(1,dtest+1)
    y_val_kfold   =  y_train[idx_val]

    # fit model on training data
    reg_dtest = LinearRegression().fit(x_train_dtest, y_train_kfold)
    
    # measure MSE on validation data
    y_hat   = reg_dtest.predict(x_val_dtest)
    mse_val = metrics.mean_squared_error(y_val_kfold, y_hat)
    r2_val  = metrics.r2_score(y_val_kfold, y_hat)
```
::: 


::: {.cell .markdown}

Notice, however, that there was a lot of wasted computation there. We computed the same polynomial features multiple times in different folds. 
Instead, we should compute the entire set of transformed features in advance, then just select the ones we need in each iteration over model order.

:::

::: {.cell .code}
```python
# create a k-fold object
nfold = 5
kf = KFold(n_splits=nfold,shuffle=True)

# model orders to be tested
dmax = 10
dtest_list = np.arange(1,dmax+1)
nd = len(dtest_list)

# create transformed features up to d_max
x_train_trans = x_train**np.arange(1,dmax+1)

for isplit, idx in enumerate(kf.split(x_train)):     
  idx_tr, idx_val = idx 

  for dtest in dtest_list:
    # get "transformed" training and validation data for this model order
    x_train_dtest =  x_train_trans[idx_tr,  :dtest]
    y_train_kfold = y_train[idx_tr]
    x_val_dtest   =  x_train_trans[idx_val, :dtest]
    y_val_kfold   = y_train[idx_val]

    # fit model on training data
    reg_dtest = LinearRegression().fit(x_train_dtest, y_train_kfold)
    
    # measure MSE on validation data
    y_hat   = reg_dtest.predict(x_val_dtest)
    mse_val = metrics.mean_squared_error(y_val_kfold, y_hat)
    r2_val  = metrics.r2_score(y_val_kfold, y_hat)
```
::: 

::: {.cell .markdown}

That's much better! Let's look at what this is doing - we'll run it again with some extra visualization:

:::


::: {.cell .code}
```python
# create a k-fold object
nfold = 5
kf = KFold(n_splits=nfold,shuffle=True)

# model orders to be tested
dmax = 10
dtest_list = np.arange(1,dmax+1)
nd = len(dtest_list)

# create transformed features up to d_max
x_train_trans = x_train**np.arange(1,dmax+1)

# create a big figure
fig, axs = plt.subplots(nfold, nd, sharex=True, sharey=True)
fig.set_figheight(nfold+1);
fig.set_figwidth(nd+1);

for isplit, idx in enumerate(kf.split(x_train)):     
  idx_tr, idx_val = idx 

  for didx, dtest in enumerate(dtest_list):
    # get "transformed" training and validation data for this model order
    x_train_dtest =  x_train_trans[idx_tr,  :dtest]
    y_train_kfold = y_train[idx_tr]
    x_val_dtest   =  x_train_trans[idx_val, :dtest]
    y_val_kfold   = y_train[idx_val]

    # fit model on training data
    reg_dtest = LinearRegression().fit(x_train_dtest, y_train_kfold)
    
    # measure MSE on validation data
    y_hat   = reg_dtest.predict(x_val_dtest)
    mse_val = metrics.mean_squared_error(y_val_kfold, y_hat)
    r2_val  = metrics.r2_score(y_val_kfold, y_hat)

    # this is just for visualization/understanding - in a "real" problem you would not include this
    p = sns.lineplot(x = x_train_dtest[:,0].squeeze(), y = reg_dtest.predict(x_train_dtest), color='red', ax=axs[isplit, didx]);
    p = sns.scatterplot(x = x_val_dtest[:, 0].squeeze(), y = y_val_kfold,  ax=axs[isplit, didx]);

plt.tight_layout()

```
::: 

::: {.cell .markdown}

Finally, we'll add some arrays in which to save the validation performance from each fold, so that we can average them afterward.

:::


::: {.cell .code}
```python
# create a k-fold object
nfold = 5
kf = KFold(n_splits=nfold,shuffle=True)

# model orders to be tested
dmax = 10
dtest_list = np.arange(1,dmax+1)
nd = len(dtest_list)

mse_val = np.zeros((nd,nfold))
r2_val  = np.zeros((nd,nfold))

# create transformed features up to d_max
x_train_trans = x_train**np.arange(1,dmax+1)

# loop over the folds
# the first loop variable tells us how many out of nfold folds we have gone through
# the second loop variable tells us how to split the data
for isplit, idx in enumerate(kf.split(x_train)):
        
  # these are the indices for the training and validation indices
  # for this iteration of the k folds
  idx_tr, idx_val = idx 

  x_train_kfold = x_train_trans[idx_tr]
  y_train_kfold = y_train[idx_tr]
  x_val_kfold = x_train_trans[idx_val]
  y_val_kfold = y_train[idx_val]

  for didx, dtest in enumerate(dtest_list):

    # get transformed features
    x_train_dtest =  x_train_kfold[:, :dtest]
    x_val_dtest   =  x_val_kfold[:, :dtest]

    # fit data
    reg_dtest = LinearRegression().fit(x_train_dtest, y_train_kfold)
    
    # measure MSE on validation data
    y_hat = reg_dtest.predict(x_val_dtest)
    mse_val[didx, isplit] = metrics.mean_squared_error(y_val_kfold, y_hat)
    r2_val[didx, isplit] = metrics.r2_score(y_val_kfold, y_hat)
```
:::

::: {.cell .markdown}

Here is the mean (across K folds) validation error for each model order:

:::

::: {.cell .code}
```python
sns.lineplot(x=dtest_list, y=mse_val.mean(axis=1), marker='o');
plt.xlabel("Model order");
plt.ylabel("K-fold MSE");
```
:::

::: {.cell .markdown}
Let's see which model order gave us the lowest MSE on the validation
data:
:::

::: {.cell .code}
```python
idx_min = np.argmin(mse_val.mean(axis=1))
d_min_mse = dtest_list[idx_min]
d_min_mse
```
:::

::: {.cell .code}
```python
mse_val.mean(axis=1)
```
:::


::: {.cell .code}
```python
sns.lineplot(x=dtest_list, y=mse_val.mean(axis=1));
sns.scatterplot(x=dtest_list, y=mse_val.mean(axis=1), hue=dtest_list==d_min_mse, s=100, legend=False);

plt.xlabel("Model order");
plt.ylabel("K-fold MSE");
```
:::

::: {.cell .markdown}
We can also select by highest R2 (instead of lowest MSE):
:::

::: {.cell .code}
```python
idx_max = np.argmax(r2_val.mean(axis=1))
d_max_r2 = dtest_list[idx_max]
d_max_r2
```
:::


::: {.cell .code}
```python
r2_val.mean(axis=1)
```
:::


::: {.cell .code}
```python
sns.lineplot(x=dtest_list, y=r2_val.mean(axis=1));
sns.scatterplot(x=dtest_list, y=r2_val.mean(axis=1), hue=dtest_list==d_max_r2, s=100, legend=False);

plt.xlabel("Model order");
plt.ylabel("K-fold R2");
```
:::


::: {.cell .markdown}
Now, we can re-fit a model of degree `d_min_mse` or `d_max_r2` on the *entire* training set:
:::

::: {.cell .code}
```python
x_train_dopt =  x_train[:, :d_max_r2]
x_test_dopt  =   x_test[:, :d_max_r2]
reg_dopt = LinearRegression().fit(x_train_dopt, y_train)
y_hat = reg_dopt.predict(x_test_dopt)
mse_dopt = metrics.mean_squared_error(y_test, y_hat)
r2_dopt  = metrics.r2_score(y_test, y_hat)

```
:::

::: {.cell .code}
```python
mse_dopt
```
:::

::: {.cell .code}
```python
r2_dopt
```
:::



::: {.cell .markdown}
### Model selection using 1-SE "rule"
:::

::: {.cell .markdown}
When using the minimum K-fold CV error for model selection, we sometimes
will still select an overly complex model
`<sup>`{=html}\[2\]`</sup>`{=html}.

As an alternative, we can use the "one standard error rule"
`<sup>`{=html}\[3\]`</sup>`{=html}. According to this "rule", we
choose the least complex model whose error is no more than one standard
error above the error of the best model - i.e. the simplest model whose
performance is comparable to the best model.

`<small>`{=html}\[2\] See [Cawley & Talbot (J of Machine Learning
Research,
2010)](http://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) for
more on this.`</small>`{=html}

`<small>`{=html}\[3\] See Chapter 7 of [Elements of Statistical
Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
`</small>`{=html}
:::

::: {.cell .markdown}
We apply this rule as follows:

-   Find the MSE for each fold for each model candidate
-   For each model candidate, compute the mean and standard error of the
    MSE over the $K$ folds. We will compute the standard error as
    $$\frac{\sigma_{\text{MSE}}}{\sqrt{K-1}}$$ where
    $\sigma_{\text{MSE}}$ is the standard deviation of the MSE over the
    $K$ folds.
-   Find the model with the smallest mean MSE (across the $K$ folds).
    Compute the *target* as mean MSE + SE for this model.
-   Select the least complex model whose mean MSE is below the target.

This works for any metric that is a "lower is better" metric. If you
are using a "higher is better" metric, such as R2, for example, you
would modify the last two steps:

-   Find the model with the **largest** mean R2 (across the $K$ folds).
    Compute the **mean R2 - SE of R2** for this model. Call this
    quantity the *target*.
-   Select the least complex model whose mean R2 is **above** the
    target.
:::

::: {.cell .code}
```python
idx_min = np.argmin(mse_val.mean(axis=1))
target = mse_val[idx_min,:].mean() + mse_val[idx_min,:].std()/np.sqrt(nfold-1)
# np.where returns indices of values where condition is satisfied
idx_one_se = np.where(mse_val.mean(axis=1) <= target)
d_one_se = np.min(dtest_list[idx_one_se])
d_one_se
```
:::

::: {.cell .code}
```python
plt.errorbar(x=dtest_list, y=mse_val.mean(axis=1), yerr=mse_val.std(axis=1)/np.sqrt(nfold-1));
plt.hlines(y=target, xmin=np.min(dtest_list), xmax=np.max(dtest_list), ls='dotted')
sns.scatterplot(x=dtest_list, y=mse_val.mean(axis=1), hue=dtest_list==d_one_se, s=100, legend=False);

plt.xlabel("Model order");
plt.ylabel("K-fold MSE");
```
:::

::: {.cell .code}
```python
idx_max = np.argmax(r2_val.mean(axis=1))
target_r2 = r2_val[idx_max,:].mean() - r2_val[idx_max,:].std()/np.sqrt(nfold-1)
# np.where returns indices of values where condition is satisfied
idx_one_se_r2 = np.where(r2_val.mean(axis=1) >= target_r2)
d_one_se_r2 = np.min(dtest_list[idx_one_se_r2])
d_one_se_r2
```
:::

::: {.cell .code}
```python
plt.errorbar(x=dtest_list, y=r2_val.mean(axis=1), yerr=r2_val.std(axis=1)/np.sqrt(nfold-1));
plt.hlines(y=target_r2, xmin=np.min(dtest_list), xmax=np.max(dtest_list), ls='dotted')
sns.scatterplot(x=dtest_list, y=r2_val.mean(axis=1), hue=dtest_list==d_one_se_r2, s=100, legend=False);

plt.xlabel("Model order");
plt.ylabel("K-fold R2");
```
:::

::: {.cell .markdown}

### Mistake: using the test set for model selection

In the examples above, we learned how to use K-fold CV to select a model using a validation set that is split off from the training data.

We understand from a previous lesson that we should not use the training set for model selection - the training error will decrease with model complexity, so this approach would always choose the most complex model (rather than one that has the best performance on data not used to fit the model).

However, you may feel tempted to use the test set for model selection (i.e. choose the model that has the best performance on the test set). But, this would be a mistake! Using the test set for model selection is a form of data leakage. If you select the model with best performance on the test set, you risk overfitting to noise in the test data, and since you have "contaminated" your test set by using it in this way, you no longer have a held-out test set on which to evaluate your final model. Your evaluation on this "contaminated" test set will be an overly optimistic evaluation.

:::

::: {.cell .markdown}

To demonstrate this, we will use an extreme example: we will generate a very small dataset with 20 samples, and a very large number of features - 10,000 features! - where

$$y = x_0 + x_1 + \epsilon$$

that is,

* the first two features, columns `0` and `1`, are relevant to the target variable
* the remaining features are just noise.

(It's an extreme example because the number of samples is very small, and the number of features is relatively so large! But this extreme example will help us illustrate our point.)

:::

::: {.cell .code}
```python
def generate_y(x):
    # y is sum of columns 0 and 1 from X + some noise
    y = x[:, 0] + x[:, 1]
    y += np.random.normal(0, .01, y.shape)
    return y

def generate_x(n):
    return np.random.uniform(0, 3, (n, 10000))

# generate data, split into training and test
X = generate_x(20)
y = generate_y(X)
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2)
```
:::

::: {.cell .markdown}

Then, we are going to train a model on only *two* features. However, assuming we do not know in advance which features are relevant and which features are not, we need to use some sort of feature selection.

First, we wil do feature selection the *wrong* way:

* For each feature, we will train a simple linear regression using only that feature.
* We will compute the R2 score of that simple linear regression on the *test* data. This will be considered the "score" of that feature.

:::

::: {.cell .code}
```python
num_features = Xtr.shape[1]
score_ts = np.zeros(num_features)
for i in tqdm(range(num_features), desc="Scoring Features"):
    # get subset of data for the feature
    Xtr_subset = Xtr[:, [i]]
    Xts_subset = Xts[:, [i]]
    # train a model on this feature
    model = LinearRegression()
    model.fit(Xtr_subset, ytr)
    # score this model using the test set
    y_pred = model.predict(Xts_subset)
    score_ts[i] = metrics.r2_score(yts, y_pred)
```
:::

::: {.cell .markdown}

Although the first two features (`0` and `1`) are the only meaningful features, they will not necessarily be the only ones with a high "score" using this method, because of the noise in the data:

:::


::: {.cell .code}
```python
plt.figure(figsize=(20,5))
plt.stem(np.arange(0, 10000),score_ts, bottom=0);
```
:::

::: {.cell .markdown}

Suppose we now find the two features with the highest test R2 score, and train a multiple regression model using those two features.

:::

::: {.cell .code}
```python
# get features with highest score
best_two_features = np.argsort(score_ts)[-2:]  # Last two (sorted in ascending order)
print(best_two_features) # Is it 0 and 1? if not... uh oh...
```
:::


::: {.cell .code}
```python
model = LinearRegression()
model.fit(Xtr[:, best_two_features], ytr)
```
:::

::: {.cell .markdown}

This model may still have a reasonably high R2 score on the test data - 

:::

::: {.cell .code}
```python
y_pred = model.predict(Xts[:, best_two_features])
print( metrics.r2_score(yts, y_pred) ) 
```
:::

::: {.cell .markdown}

But, that doesn't mean that it will really do well at making predictions for new data not used at all in the training process - let's generate some new data using the same functions, and try it out:

:::

::: {.cell .code}
```python
X_new = generate_x(20)
y_new = generate_y(X_new)
```
:::


::: {.cell .code}
```python
# on actual new data, score is very low
y_new_pred = model.predict(X_new[:, best_two_features])
print( metrics.r2_score(y_new, y_new_pred) )
```
:::


::: {.cell .markdown}

This approach, in which we used the test set for model selection, had a major problem: although the model we selected was actually *not* a good model, the evaluation was "overly optimistic" - it implied that the model *was* good. 

This is because we "contaminated" the test set by using it for model selection. Let's try again with the correct approach, where we use a separate validation set split off from the training data for model selection:

:::


::: {.cell .code}
```python

X_train, X_val, y_train, y_val = train_test_split(Xtr, ytr, test_size=4, random_state=2)

# "Score" each feature by fitting on the training set and evaluating on the test set
num_features = X_train.shape[1]
score_vl = np.zeros(num_features)
for i in tqdm(range(num_features), desc="Scoring Features"):
    # get subset of data for the feature
    Xtr_subset = X_train[:, [i]]
    Xvl_subset = X_val[:, [i]]
    # train a model on this feature
    model = LinearRegression()
    model.fit(Xtr_subset, y_train)
    # score this model using the test set
    y_pred = model.predict(Xvl_subset)
    score_vl[i] = metrics.r2_score(y_val, y_pred)

```
:::



::: {.cell .code}
```python
# get features with highest score
best_two_features = np.argsort(score_vl)[-2:]  # Last two (sorted in ascending order)
print(best_two_features) # Is it 0 and 1? if not... uh oh...
```
:::

::: {.cell .code}
```python
model = LinearRegression()
model.fit(Xtr[:, best_two_features], ytr)
```
:::

::: {.cell .markdown}

When we evaluate *this* model on the test set:

:::

::: {.cell .code}
```python
y_pred = model.predict(Xts[:, best_two_features])
print( metrics.r2_score(yts, y_pred) ) 
```
:::

::: {.cell .markdown}

we understand (correctly!) that the model is not useful. The evaluation on the "clean" held-out test set is similar to the model performance on really "new" data:

:::


::: {.cell .code}
```python
y_new_pred = model.predict(X_new[:, best_two_features])
print( metrics.r2_score(y_new, y_new_pred) )

```
:::

::: {.cell .markdown}

This example highlights the danger of using the test set in any meaningful way before the final evaluation (including training, data processing, or model selection) - with a "contaminated" test set, our evaluation may be overly optimistic and we will not understand the true performance of our model.

Using the test set for model selection is not the only mistake that can lead to data leakage and an overly optimistic evaluation, though! The next section describes another...

:::



::: {.cell .markdown}

## Predicting the course of COVID with a “cubic model”

As part of the materials for this lesson, you read about some attempts early in the COVID-19 pandemic to predict how the number of cases or deaths would evolve.  You were asked to consider:

> The forecasts produced by these models were all very wrong, but they appeared to fit the data well! What was wrong with the approach used to produce these models? How did they miscalculate so badly?

Now, we'll take that process apart, see what went wrong, and see what we could have done differently.

:::

::: {.cell .markdown}
First, we will get U.S. COVID data and read it in to our notebook environment. We'll also add a field called `daysElapsed` which will count the number of days since March 1, 2020 for each row of data.
:::


::: {.cell .code}
```python
!wget https://covidtracking.com/data/download/national-history.csv -O national-history.csv
```
:::

::: {.cell .code}
```python
df = pd.read_csv('national-history.csv')
df.date = pd.to_datetime(df.date)
df = df.sort_values(by="date")
df = df.assign(daysElapsed =  (df.date - pd.to_datetime('2020-03-01')).dt.days)
df.head()
```
:::

:::

::: {.cell .markdown}

Let's assume that we are making this prediction sometime near the beginning of May 2020 (like in the reading), well into the first wave in the U.S., and we want to predict when this wave will fall off (deaths go back to zero). 

We'll use all of the data up to May 2020 for training, since that is what is available at "training time". But afterwards, we'll go back and see how well our model did by comparing its predictions for May and June 2020 to the real course of the pandemic.

:::


::: {.cell .code}
```python
df_tr = df[(df.date <= '2020-05-01') & (df.date > '2020-03-01')]
df_ts = df[(df.date < '2020-06-30') & (df.date >= '2020-05-01')]
```
:::

::: {.cell .code}
```python
sns.scatterplot(x=df_tr.daysElapsed, y=df_tr.deathIncrease);
```
:::

::: {.cell .markdown}

Furthermore, we will use a polynomial basis to fit a linear regression model, so let's generate "transformed" versions of our feature: days since the beginning of March 2020.
:::

::: {.cell .code}
```python
df_tr_poly = df_tr.daysElapsed.values.reshape(-1,1)**np.arange(1, 5)
df_ts_poly = df_ts.daysElapsed.values.reshape(-1,1)**np.arange(1, 5)
```
:::


::: {.cell .markdown}
Let's now go ahead and fit a model to our training data, then compute the R2 score for the model. It seems like a good fit on the training data:
:::


::: {.cell .code}
```python
reg_covid = LinearRegression().fit(df_tr_poly, df_tr.deathIncrease)
deathIncrease_fitted = reg_covid.predict(df_tr_poly)
metrics.r2_score(df_tr.deathIncrease, deathIncrease_fitted)
```
:::


::: {.cell .markdown}
and looks reasonably good in this visualization (although of course, we recognize that negative deaths should be impossible):
:::


::: {.cell .code}
```python
sns.scatterplot(x=df_tr.daysElapsed, y=df_tr.deathIncrease);
sns.lineplot(x=df_tr.daysElapsed, y=deathIncrease_fitted);
```
:::


::: {.cell .markdown}
But despite looking good on training data, the model does a terrible job of predicting the end of the first wave. The model predicts that deaths fall to zero around day 70, but we can see that the actual fall off is much slower, and there are still hundreds of deaths each day at day 120.
:::


::: {.cell .code}
```python
deathIncrease_fitted_future = reg_covid.predict(df_ts_poly)

sns.scatterplot(x=df_tr.daysElapsed, y=df_tr.deathIncrease);
sns.lineplot(x=df_tr.daysElapsed, y=deathIncrease_fitted);

sns.scatterplot(x=df_ts.daysElapsed, y=df_ts.deathIncrease);
sns.lineplot(x=df_ts.daysElapsed, y=deathIncrease_fitted_future);

plt.ylim(0, 3000);
```
:::

::: {.cell .markdown}

and the R2 score shows that this model has very poor performance on the test set of future data:

:::


::: {.cell .code}
```python
metrics.r2_score(df_ts.deathIncrease, deathIncrease_fitted_future)
```
:::


::: {.cell .markdown}

The R2 score on the training set didn't tell us about the model's performance on its "true" task: making predictions about the future. This is a "worst case scenario" for model evaluation:

* **Best case**: Model does well in evaluation, model does well when deployed for its "true" task.
* **Second best case**: Model does poorly in evaluation, but at least you know not to deploy the model for its "true" task.
* **Worst case**: Model does well in evaluation, you deploy the model for its "true" task, and then the model does very poorly - in this case you make bad decisions, lose credibility, etc. because you trust the model output. You would be much better off if you understood from the evaluation that the model is not useful.

:::


::: {.cell .markdown}

It's likely that the examples in the "Predicting the course of COVID with a “cubic model”" case study only evaluated their models on the same data used for training, which is why they vastly overestimated its ability. But, even if they had tried to evaluate on a held-out validation set, if they had not split the data in a way that respects its structure, they would *still* vastly overestimate the capabilities of the model.

Let's see how. Since this is a small dataset, instead of a single train/test split, we'll evaluate the model using KFold CV with five splits, and we'll report the average R2 score on the "held-out" set across folds.

(Note that in this case, we are using K-fold CV only for model evaluation, not model selection.)

:::


::: {.cell .code}
```python
nfold = len(idxval)
kf = KFold(shuffle=True, n_splits=nfold)

r2_badcv = np.zeros(nfold)

fig, axs = plt.subplots(1, nfold, sharex=True, sharey=True)
fig.set_figwidth(15);
fig.set_figheight(2);


for isplit, (train_idx, val_idx) in enumerate(kf.split(df_tr_poly)):
    x_train_kfold, x_val_kfold = df_tr_poly[train_idx], df_tr_poly[val_idx]
    y_train_kfold, y_val_kfold = df_tr.deathIncrease.iloc[train_idx], df_tr.deathIncrease.iloc[val_idx]

    model = LinearRegression().fit(x_train_kfold, y_train_kfold)
    y_pred = model.predict(x_val_kfold)
    r2_badcv[isplit] = metrics.r2_score(y_val_kfold, y_pred)

    _ = sns.scatterplot(x=x_train_kfold[:, 0], y=y_train_kfold, ax=axs[isplit]);
    _ = sns.scatterplot(x=x_val_kfold[:, 0], y=y_val_kfold, ax=axs[isplit]);
    _ = axs[isplit].set_title(f"Fold {isplit+1}");

```
:::

::: {.cell .code}
```python
r2_badcv.mean()
```
:::


::: {.cell .markdown}

When the model makes predictions on the "validation" set in the example above, it has already been trained on the data points immediately before and after each validation sample.

This is a much easier task than the "true" task that the model will perform - making predictions for a sequence of consecutive future dates.

To evaluate the performance of the model, we should make sure that the validation task mimics this "true" task. 

One possible approach would be to create multiple "folds", where in each fold, the number of samples in the training set increases (and the validation set is always the ten days after the training set - we are validating whether our model can predict deaths ten days into the future).

Here's what that might look like (blue dots are training data, orange dots are validation data):

:::


::: {.cell .code}
```python
nfold = len(idxval)
tscv = TimeSeriesSplit(n_splits=nfold, test_size=10)

r2_tscv = np.zeros(nfold)

fig, axs = plt.subplots(1, nfold, sharex=True, sharey=True)
fig.set_figwidth(15);
fig.set_figheight(2);


for isplit, (train_idx, val_idx) in enumerate(tscv.split(df_tr_poly)):
    x_train_kfold, x_val_kfold = df_tr_poly[train_idx], df_tr_poly[val_idx]
    y_train_kfold, y_val_kfold = df_tr.deathIncrease.iloc[train_idx], df_tr.deathIncrease.iloc[val_idx]

    model = LinearRegression().fit(x_train_kfold, y_train_kfold)
    y_pred = model.predict(x_val_kfold)
    r2_tscv[isplit] = metrics.r2_score(y_val_kfold, y_pred)

    _ = sns.scatterplot(x=x_train_kfold[:, 0], y=y_train_kfold, ax=axs[isplit]);
    _ = sns.scatterplot(x=x_val_kfold[:, 0], y=y_val_kfold, ax=axs[isplit]);
    _ = axs[isplit].set_title(f"Fold {isplit+1}");
```
:::


::: {.cell .markdown}
With this validation approach, we find (correctly) that the polynomial model is *not* good at predicting COVID cases:
:::

::: {.cell .code}
```python
r2_tscv.mean()
```
:::


::: {.cell .markdown}

This case study highlights, the importance of evaluating on a held-out test set not used for training, but also, of making sure that the evaluation task is comparable to the "true" task that the model will be used for! 

In the evaluation with shuffled data, the R2 score seems to be high, but this is only due to data leakage. The evaluation is not valid, and the model is really not useful. The correct evaluation, with a time series split, shows that the model does not have any predictive benefit.

:::

