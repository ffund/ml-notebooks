---
title: 'Bias and variance of SVMs'
author: 'Fraida Fund'
jupyter:
  colab:
    name: '7-svm-bias-variance.ipynb'
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Bias and variance of SVMs

*Fraida Fund*
:::

::: {.cell .markdown }
In this notebook, we will explore the bias and variance of SVM models,
and see how we can tune this tradeoff.
:::

::: {.cell .code }
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from sklearn.datasets import make_blobs
```
:::

::: {.cell .markdown }
### Regularization level
:::

::: {.cell .markdown }
Suppose we want to train a model to classify two "blobs" of data.
:::

::: {.cell .code }
```python
n_repeat = 100
n_test = 500
n_train = 100
sigma= 0.8
cluster_centers = np.array([[-1,1],[2,2]])
```
:::

::: {.cell .code }
```python
y_predict = np.zeros((n_test, n_repeat, 2))
```
:::

::: {.cell .code }
```python
x_test, y_test = make_blobs(n_samples=n_test, centers=cluster_centers,
                  random_state=0, cluster_std=sigma)
```
:::

::: {.cell .code }
```python
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test);

plt.xlabel("x1");
plt.ylabel("x2");
plt.xlim(-5,5);
plt.ylim(-2,5);

# get the true decision boundary 
mid = [cluster_centers[:,0].mean(), cluster_centers[:,1].mean()]
slp = -1.0/((cluster_centers[1,1]-cluster_centers[0,1])/(cluster_centers[1,0]-cluster_centers[0,0]))
b = mid[1]-slp*mid[0]
x_true = np.arange(-5,5)
y_true = slp*x_true + b
sns.lineplot(x=x_true, y=y_true, color='black', label="True decision boundary")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
```
:::

::: {.cell .markdown }
Which will have greater bias, and which will have greater variance?

-   **Model A**: Linear SVM with $C=0.01$

-   **Model B**: Linear SVM with $C=100$
:::

::: {.cell .markdown }
Note: here is $C$ in the SVM problem:

$$
\begin{aligned}
\operatorname*{minimize}_{\mathbf{w}, \mathbf{\epsilon}} \quad & \frac{1}{2} \sum_{j=1}^p w_j^2  + C \sum_{i=1}^n \epsilon_i  \\
\text{subject to} \quad & y_i(w_0 + \sum_{j=1}^p w_j x_{ij}) \geq 1-\epsilon_i, \quad \forall i \\
& \epsilon_i \geq 0, \quad \forall i
\end{aligned}
$$

The greater the value of $C$, the more heavily the "margin violators"
penalize the overall objective function. Therefore,

-   If $C$ is large, the margin must be narrow (with few "margin
    violators").
-   If $C$ is small, the margin may be wider (with more "margin
    violators").
:::

::: {.cell .code }
```python
Z_sim = np.zeros((40000, n_repeat, 2))

fig = plt.figure(figsize=(12,4))
ax_a, ax_b = fig.subplots(1, 2, sharex=True, sharey=True)

# now simulate training the model many times, on different training data every time
# and evaluate using the test data
for i in tqdm(range(n_repeat), total=n_repeat, desc="Simulation iteration"):
 
  # train both models on newly generated training data
  X, y = make_blobs(n_samples=n_test, centers=cluster_centers,
                  cluster_std=sigma)

  clf_a = SVC(kernel='linear', C=0.01).fit(X, y)
  clf_b = SVC(kernel='linear', C=100.0).fit(X, y)

  y_predict[:, i, 0] = clf_a.predict(x_test)
  y_predict[:, i, 1] = clf_b.predict(x_test)


  xx, yy = np.meshgrid(np.arange(-5, 5, .05),
                      np.arange(-5, 5, .05))
  
  Z = clf_a.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 0] = Z
  Z = Z.reshape(xx.shape)
  ax_a.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');

  plt.xlim(-5,5);
  plt.ylim(-2,5);

  Z = clf_b.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 1] = Z
  Z = Z.reshape(xx.shape)
  ax_b.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');

  plt.xlim(-5,5);
  plt.ylim(-2,5);


cs_a = ax_a.contour(xx, yy, Z_sim[:,:,0].mean(axis=1).reshape(200,200), levels=[0.5], colors='magenta', linewidths=2);
cs_b = ax_b.contour(xx, yy, Z_sim[:,:,1].mean(axis=1).reshape(200,200), levels=[0.5], colors='magenta', linewidths=2);

# plot data
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, ax=ax_a, legend=False);
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, ax=ax_b, legend=False);

sns.lineplot(x=x_true, y=y_true, color='black', ax=ax_a)
sns.lineplot(x=x_true, y=y_true, color='black', ax=ax_b)


ax_a.set_title("Model A");
ax_b.set_title("Model B");

ax_a.set_ylabel("x2");
ax_a.set_xlabel("x1");
ax_b.set_xlabel("x1");
```
:::

::: {.cell .markdown }
### Kernels
:::

::: {.cell .code }
```python
def generate_polynomial_classifier_data(n=100, xrange=[-1,1], coefs=[1,0.5,0,2], sigma=0.5):
  x = np.random.uniform(xrange[0], xrange[1], size=(n, 2))
  ysep = np.polynomial.polynomial.polyval(x[:,0],coefs)
  y = (x[:,1]>ysep).astype(int)
  x[:,0] = x[:,0] + sigma * np.random.randn(n)
  x[:,1] = x[:,1] + sigma * np.random.randn(n)
  return x, y
```
:::

::: {.cell .code }
```python
n_repeat = 100
n_test = 500
n_train = 1000
sigma= 0.3
coefs=np.array([0.3, 1, -1.5, -2])
xrange=[-1,1]
```
:::

::: {.cell .code }
```python
y_predict = np.zeros((n_test, n_repeat, 2))
```
:::

::: {.cell .code }
```python
# generate test data once
x_test, y_test = generate_polynomial_classifier_data(n=n_test, xrange=xrange, coefs=coefs, sigma=sigma)
```
:::

::: {.cell .code }
```python
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test);

plt.xlabel("x1");
plt.ylabel("x2");
plt.xlim((xrange[0], xrange[1]));
plt.ylim((xrange[0], xrange[1]));

# Plot true function
xtrue = np.arange(-1.5, 1.5, .05)
ytrue = np.polynomial.polynomial.polyval(xtrue,coefs) 
sns.lineplot(x=xtrue, y=ytrue, color='black', label='True decision boundary');


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
```
:::

::: {.cell .markdown }
Suppose we want to train a model to classify data that is separated by a
polynomial boundary.
:::

::: {.cell .markdown }
Which will have greater bias, and which will have greater variance?

-   **Model A**: SVM with linear kernel, $C = 1$

-   **Model B**: SVM with RBF kernel, $C = 1$
:::

::: {.cell .code }
```python
Z_sim = np.zeros((3600, n_repeat, 2))

fig = plt.figure(figsize=(12,4))
ax_a, ax_b = fig.subplots(1, 2, sharex=True, sharey=True)

# now simulate training the model many times, on different training data every time
# and evaluate using the test data
for i in tqdm(range(n_repeat), total=n_repeat, desc="Simulation iteration"):

  # train both models on newly generated training data
  X, y = generate_polynomial_classifier_data(n=n_train, xrange=xrange, coefs=coefs, sigma=sigma)

  clf_a = SVC(kernel='linear', C=1).fit(X, y)
  clf_b = SVC(kernel='rbf', gamma=10,  C=1).fit(X, y)

  y_predict[:, i, 0] = clf_a.predict(x_test)
  y_predict[:, i, 1] = clf_b.predict(x_test)


  xx, yy = np.meshgrid(np.arange(-1.5, 1.5, .05),
                      np.arange(-1.5, 1.5, .05))
  
  Z = clf_a.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 0] = Z
  Z = Z.reshape(xx.shape)
  ax_a.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');
  plt.xlim((xrange[0], xrange[1]));
  plt.ylim((xrange[0], xrange[1]));

  Z = clf_b.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 1] = Z
  Z = Z.reshape(xx.shape)
  ax_b.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');
  plt.xlim((xrange[0], xrange[1]));
  plt.ylim((xrange[0], xrange[1]));


cs_a = ax_a.contour(xx, yy, Z_sim[:,:,0].mean(axis=1).reshape(60,60), levels=[0.5], colors='magenta', linewidths=2);
cs_b = ax_b.contour(xx, yy, Z_sim[:,:,1].mean(axis=1).reshape(60,60), levels=[0.5], colors='magenta', linewidths=2);


# Plot true function
xtrue = np.arange(-1.5, 1.5, .05)
ytrue = np.polynomial.polynomial.polyval(xtrue,coefs) 
sns.lineplot(x=xtrue, y=ytrue, color='black', ax=ax_a);
sns.lineplot(x=xtrue, y=ytrue, color='black', ax=ax_b);

sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, ax=ax_a, legend=False, alpha=0.1);
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, ax=ax_b, legend=False, alpha=0.1);

ax_a.set_title("Model A");
ax_b.set_title("Model B");

ax_a.set_ylabel("x2");
ax_a.set_xlabel("x1");
ax_b.set_xlabel("x1");
```
:::

::: {.cell .markdown }
### RBF parameter
:::

::: {.cell .markdown }
Recall that the RBF kernel is defined as

$$K(x,z) = \exp(-\frac{\|x-z\|^2_2}{\sigma^2})$$

where $\sigma$ is the bandwidth, or equivalently, using a $\gamma$
parameter,

$$K(x, z) = \exp(-\gamma \| x - z \|^2_2 )$$
:::

::: {.cell .markdown }
For example, here is the RBF kernel centered on a single point, computed
over the entire feature space, with two different values of $\gamma$:
:::

::: {.cell .code }
```python
from sklearn.metrics.pairwise import rbf_kernel

test_point = np.random.uniform(0,1,size=2)
xx, yy = np.meshgrid(np.arange(0 ,5, .05), np.arange(0, 5, .05))

gamma_a=0.05
gamma_b=5
Z_a = rbf_kernel(np.c_[xx.ravel(), yy.ravel()], test_point.reshape(1, -1), gamma=gamma_a)
Z_b = rbf_kernel(np.c_[xx.ravel(), yy.ravel()], test_point.reshape(1, -1), gamma=gamma_b)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

plt.scatter(x=test_point[0], y=test_point[1])
cs = plt.contourf(xx, yy, Z_a.reshape(xx.shape), vmin=0, vmax=1);
plt.title("Gamma: %f" % gamma_a);

plt.subplot(1,2,2)


plt.scatter(x=test_point[0], y=test_point[1])
cs = plt.contourf(xx, yy, Z_b.reshape(xx.shape), vmin=0, vmax=1);
plt.title("Gamma: %f" % gamma_b);

plt.subplots_adjust(right=0.8);
cbar_ax = plt.axes([0.85, 0.15, 0.05, 0.7]);
plt.colorbar(cax=cbar_ax);
```
:::

::: {.cell .markdown }
We can see that when the kernel bandwidth is large ($\gamma$ is small),
the influence of each point extends much farther in the feature space
than if the kernel bandwidth is small ($\gamma$ is large).
:::

::: {.cell .markdown }
Suppose we want to train a model to classify data that is separated by a
polynomial boundary.

Which will have greater bias, and which will have greater variance?

-   **Model A**: SVM with RBF kernel and $\gamma=0.05$

-   **Model B**: SVM with RBF kernel and $\gamma=5$
:::

::: {.cell .code }
```python
n_repeat = 100
n_test = 500
n_train = 100
sigma= 0.3
coefs=np.array([0.3, 1, -1.5, -2])
xrange=[-1,1]
```
:::

::: {.cell .code }
```python
y_predict = np.zeros((n_test, n_repeat, 2))
```
:::

::: {.cell .code }
```python
# generate test data once
x_test, y_test = generate_polynomial_classifier_data(n=n_test, xrange=xrange, coefs=coefs, sigma=sigma)
```
:::

::: {.cell .code }
```python
Z_sim = np.zeros((3600, n_repeat, 2))

fig = plt.figure(figsize=(12,4))
ax_a, ax_b = fig.subplots(1, 2, sharex=True, sharey=True)

# now simulate training the model many times, on different training data every time
# and evaluate using the test data
for i in tqdm(range(n_repeat), total=n_repeat, desc="Simulation iteration"):

  # train both models on newly generated training data
  X, y = generate_polynomial_classifier_data(n=n_train, xrange=xrange, coefs=coefs, sigma=sigma)

  clf_a = SVC(kernel='rbf', gamma=0.05, C=10).fit(X, y)
  clf_b = SVC(kernel='rbf', gamma=5, C=10).fit(X, y)

  y_predict[:, i, 0] = clf_a.predict(x_test)
  y_predict[:, i, 1] = clf_b.predict(x_test)


  xx, yy = np.meshgrid(np.arange(-1.5, 1.5, .05),
                      np.arange(-1.5, 1.5, .05))
  
  Z = clf_a.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 0] = Z
  Z = Z.reshape(xx.shape)
  ax_a.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');
  plt.xlim((xrange[0], xrange[1]));
  plt.ylim((xrange[0], xrange[1]));

  Z = clf_b.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_sim[:, i, 1] = Z
  Z = Z.reshape(xx.shape)
  ax_b.contour(xx, yy, Z, levels=[0.5], alpha=0.1, colors='plum');
  plt.xlim((xrange[0], xrange[1]));
  plt.ylim((xrange[0], xrange[1]));


cs_a = ax_a.contour(xx, yy, Z_sim[:,:,0].mean(axis=1).reshape(60,60), levels=[0.5], colors='magenta', linewidths=2);
cs_b = ax_b.contour(xx, yy, Z_sim[:,:,1].mean(axis=1).reshape(60,60), levels=[0.5], colors='magenta', linewidths=2);


# Plot true function
xtrue = np.arange(-1.5, 1.5, .05)
ytrue = np.polynomial.polynomial.polyval(xtrue,coefs) 
sns.lineplot(x=xtrue, y=ytrue, color='black', ax=ax_a);
sns.lineplot(x=xtrue, y=ytrue, color='black', ax=ax_b);

#sns.scatterplot(x_test[:,0], x_test[:,1], y_test, ax=ax_a, legend=False, alpha=0.1);
#sns.scatterplot(x_test[:,0], x_test[:,1], y_test, ax=ax_b, legend=False, alpha=0.1);

ax_a.set_title("Model A");
ax_b.set_title("Model B");

ax_a.set_ylabel("x2");
ax_a.set_xlabel("x1");
ax_b.set_xlabel("x1");
```
:::

::: {.cell .markdown }
## Hyperparameter search
:::

::: {.cell .markdown }
For models with a single hyperparameter controlling bias-variance (for
example: $k$ in $k$ nearest neighbors), we used sklearns\'s `KFoldCV` or
`validation_curve` to test a range of values for the hyperparameter, and
to select the best one.

When we have *multiple* hyperparameters to tune, we can use
`GridSearchCV` to select the best *combination* of them.

For example, we just saw saw three ways to tune the bias-variance of an
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
Now we'l set up the grid search. We can use `fit` on it, just like any
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
we'l get better results with smaller values of $\gamma$. For smaller
values of $C$, we expect that we'l get better results with larger
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
clf.fit(X_train, y_train)
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
