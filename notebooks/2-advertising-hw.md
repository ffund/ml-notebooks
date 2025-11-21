---
title:  'Linear regression on the Advertising data'
author: 'Fraida Fund'
---

::: {.cell .markdown}

# Assignment: Linear regression on the Advertising data


_Fraida Fund_

:::

::: {.cell .markdown}

Fill in your name and net ID:

* **Name**:
* **net ID**:

:::

::: {.cell .markdown}


Make a copy of this notebook in your own Google Drive and, as you work through it, fill in missing code and answers to the questions.

After you are finished, you will copy your answers from individual sections *and* a copy of the entire notebook into PrairieLearn for submission. (Note that the PrairieLearn autograder will expect that you have used exactly the variable names shown in this template notebook.)

Answers to open-ended questions (e.g. "Comment on the results..") must be **in your own words**, reflecting your own interpretation and understanding, and must refer to specific results (including numeric values) you obtained in this notebook.

:::


::: {.cell .markdown}

To illustrate principles of linear regression, we are going to use some data from the textbook "An Introduction to Statistical Learning" (Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani).

The dataset is described as follows:

> Suppose that we are statistical consultants hired by a client to
> provide advice on how to improve sales of a particular product. The 
> `Advertising` data set consists of the sales of that product in 200
> different markets, along with advertising budgets for the product in
> each of those markets for three different media: TV, radio, and
> newspaper. 
>
> ... 
>
> It is not possible for our client to directly increase
> sales of the product. On the other hand, they can control the
> advertising expenditure in each of the three media. Therefore, if we
> determine that there is an association between advertising and sales,
> then we can instruct our client to adjust advertising budgets, thereby
> indirectly increasing sales. In other words, our goal is to develop an
> accurate model that can be used to predict sales on the basis of the
> three media budgets.

Sales are reported in thousands of units, and TV, radio, and newspaper
budgets, are reported in thousands of dollars.

For this assignment, you will fit a linear regression model to a small dataset. You will iteratively improve your linear regression model by examining the residuals at each stage, in order to identify problems with the model.

:::

::: {.cell .code}
``` {.python}
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
```
:::

::: {.cell .markdown}

### 0. Read in and pre-process data

In this section, you will read in the "Advertising" data, and make sure it is loaded correctly.  Then, split the data into training data (70%) and test data (30%). We will use *only* the training data in our visualizations, etc.

Visually inspect the data using a pairplot, and note any meaningful observations. In particular, comment on which features appear to be correlated with product sales, and which features appear to be correlated with one another. 

**The code in this section is provided for you**. 

:::


::: {.cell .markdown}
#### Read in data
:::


::: {.cell .code}
```python
!wget 'https://www.statlearning.com/s/Advertising.csv' -O 'Advertising.csv'
```
:::


::: {.cell .code}
```python
df  = pd.read_csv('Advertising.csv', index_col=0)
df.head()
```
:::

::: {.cell .markdown}
Note that in this dataset, the first column in the data file is the row
label; that's why we use `index_col=0` in the `read_csv` command. If we
would omit that argument, then we would have an additional (unnamed)
column in the dataset, containing the row number.

(You can try removing the `index_col` argument and re-running the cell
above, to see the effect and to understand why we used this argument.)
:::


::: {.cell .markdown}
#### Split up data


We will use 70% of the data for training and the remaining 30% as a held-out test set to evaluate
the regression model on data *not* used for training.

:::

::: {.cell .code}
```python
train, test = train_test_split(df, test_size=0.3, random_state=9)
```
:::

::: {.cell .markdown}

We set the `random_state` to a constant so that every time you run this notebook, exactly the same data points will be assigned to test vs. training sets.  This is helpful in the debugging stage. 

:::


::: {.cell .code}
```python
train.info()
```
:::

::: {.cell .code}
```python
test.info()
```
:::



::: {.cell .markdown}

#### Visually inspect the data

:::


::: {.cell .code}
```python
sns.pairplot(train);
```
:::

::: {.cell .markdown}

The most important panels here are on the bottom row, where `sales` is
on the vertical axis and the advertising budgets are on the horizontal
axes. 

Looking at this row, we may identify some features that appear to be useful 
predictive features for `sales`, but we cannot *rule out* any features based on this visualization.

:::

::: {.cell .markdown}

**Comment on this plot**. What features appear to be related to the target variable? What features appear to be correlated with other features?

:::



::: {.cell .markdown}


### 1. Fit simple linear regression models

Use the training data to fit a simple linear regression to predict product sales, for each of three features: TV ad budget, radio ad budget, and newspaper ad budget. 

In other words, you will fit *three* regression models, 

* a `reg_tv` model that uses `tv` as a feature to predict `sales`
* a `reg_radio` model that uses `radio` as a feature to predict `sales`
* a `reg_news` model that uses `newspaper` as a feature to predict `sales`

:::

::: {.cell .markdown}
#### Fit a simple linear regression

The code for the first model, `reg_tv`, is provided for you. Fill in `reg_radio` and `reg_news`.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

reg_tv    = LinearRegression().fit(train[['TV']], train['sales'])
# reg_radio = ...
# reg_news  = ...
```
:::

::: {.cell .markdown}

#### Look at coefficients

Look at the intercept $w_0$ and the slope coefficient $w_1$ of each model. The code for the first model, `reg_tv`, is provided for you.

:::

::: {.cell .code}
```python
print("TV       : ", reg_tv.coef_[0], reg_tv.intercept_)
# print("Radio    : ",  ??? )
# print("Newspaper: ",  ??? )
```
:::


::: {.cell .markdown}
#### Plot data and regression line

The following cell will show a visualization of the training data and the regression model you fitted in each case. 

Note that the range of the horizontal axis in each case is very different, because the upper end of the range of spending on TV ads is much larger than e.g. on radio ads. Therefore, you can't judge the relative slope just by looking at the visualization - you will have to refer back to the slop coefficients you printed above.

:::

::: {.cell .code}
```python
fig = plt.figure(figsize=(12,3))

plt.subplot(1,3,1)
sns.scatterplot(data=train, x="TV", y="sales");
sns.lineplot(data=train, x="TV", y=reg_tv.predict(train[['TV']]), color='red');

plt.subplot(1,3,2)
sns.scatterplot(data=train, x="radio", y="sales");
sns.lineplot(data=train, x="radio", y=reg_radio.predict(train[['radio']]), color='red');

plt.subplot(1,3,3)
sns.scatterplot(data=train, x="newspaper", y="sales");
sns.lineplot(data=train, x="newspaper", y=reg_news.predict(train[['newspaper']]), color='red');
```
:::

::: {.cell .markdown}

#### Compute R2, MSE for simple regression

For each model, let's get:

* the predictions on the training set: `y_pred_tr_tv`, `y_pred_tr_radio`, `y_pred_tr_news`
* the R2 score on the training set: `r2_tr_tv`, `r2_tr_radio`, `r2_tr_news`
* the MSE on the training set: `mse_tr_tv`, `mse_tr_radio`, `mse_tr_news`

These are already provided for you, for the TV case.

:::


::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

y_pred_tr_tv    = reg_tv.predict(train[['TV']])
# y_pred_tr_radio = ...
# y_pred_tr_news  = ...
```
:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

r2_tr_tv    = metrics.r2_score(train['sales'], y_pred_tr_tv)
# r2_tr_radio = ...
# r2_tr_news  = ...
```
:::

::: {.cell .code}
```python
print("TV       : ", r2_tr_tv)
# print("Radio    : ", r2_tr_radio)
# print("Newspaper: ", r2_tr_news)
```
:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

mse_tr_tv    = metrics.mean_squared_error(train['sales'], y_pred_tr_tv)
# mse_tr_radio = ...
# mse_tr_news  = ...
```
:::

::: {.cell .code}
```python
print("TV       : ", mse_tr_tv)
# print("Radio    : ", mse_tr_radio)
# print("Newspaper: ", mse_tr_news)
```
:::





::: {.cell .markdown}

### 2. Explore the residuals for the single linear regression models

Computing MSE or R2 is not sufficient to diagnose a problem with a linear regression. 

In this section, you will create some additional visualizations of the training data as described below. to help you identify any problems with the regression. 

:::


::: {.cell .markdown}

#### Compute residuals

First, for each of the three regression models, you will compute the residuals ($y - \hat{y}$). 

The code for the TV model is already provided for you.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

residual_tv_tr = train['sales'] - reg_tv.predict(train[['TV']])
# residual_news_tr = ...
# residual_radio_tr = ...
```
:::

::: {.cell .markdown}

#### Plot predicted vs. actual sales

Next, you'll create a plot of predicted sales vs. actual sales for each of the three models.

 You will organize these as three *subplots* in one row. In each subplot, for a different model,

* Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis) and that all three subplots use the same scale. Since the units of `sales` and predicted `sales` should be the same, if we use the same scale we can make direct comparisons based on the appearance of the plots. Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area for each subplot should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes, and each plot. 

Code is provided for you for the TV model - add subplots to the figure to also show the radio model and the newspaper model.

:::

::: {.cell .code}
```python

fig = plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.scatterplot(data=train, x="sales", y=reg_tv.predict(train[['TV']]));
plt.ylabel('Predicted sales');
plt.ylim(0,30);
plt.xlim(0,30);
plt.title("Regression on TV");

```
:::

::: {.cell .markdown}

**Comment on this plot**. What would you expect this plot to look like for a model that explains the data well? What does the plot tell you about your models in this specific case?

:::

::: {.cell .markdown}

#### Plot residuals vs. actual sales

Next, you'll create a plot of residuals vs. actual sales for each of the three models.

 You will organize these as three *subplots* in one row. In each subplot, for a different model,

* Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Use the same vertical scale for all three subplots, and the same horizontal scale for all three subplots (but the vertical scale and the horizontal scale will not be the same as one another!).  Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes, and each plot. 

Code is provided for you for the TV model - add subplots to the figure to also show the radio model and the newspaper model.

:::

::: {.cell .code}
```python

fig = plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.scatterplot(x=train['sales'], y=residual_tv_tr);
plt.xlabel('Actual Sales')
plt.ylabel('Residual');
plt.ylim(-20, 20)
plt.title("Regression on TV");

```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales?

:::


::: {.cell .markdown}

#### Plot residuals vs. features

Finally, you'll create a plot of residuals vs. features, for each combination of "model" and "feature" (9 subplots total).

In each column, for a different model,


* create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis, for each feature. 
* Use the same vertical scale for all subplots (but the horizontal scale will depend on the feature, and will be different for each feature! The upper end of the TV ad spending range is different from the radio ad spending range.)  Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area for each subplot should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Make sure to clearly label each axis, and also label each subplot with a title that indicates which regression model it uses. 

Code is provided for you for the TV model - add subplots to the figure to also show the radio model and the newspaper model.

:::

::: {.cell .code}
```python

plt.figure(figsize=(13,12))

plt.subplot(3,3,1)
sns.scatterplot(x=train['TV'], y=residual_tv_tr);
plt.xlabel('TV ad $')
plt.ylabel('Residual');
plt.ylim(-20, 20);
plt.title("Regression on TV");

plt.subplot(3,3,4)
sns.scatterplot(x=train['radio'], y=residual_tv_tr);
plt.xlabel('Radio ad $')
plt.ylabel('Residual');
plt.ylim(-20, 20);

plt.subplot(3,3,7)
sns.scatterplot(x=train['newspaper'], y=residual_tv_tr);
plt.xlabel('Newspaper ad $')
plt.ylabel('Residual');
plt.ylim(-20, 20);

plt.tight_layout();
```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features?


:::


::: {.cell .markdown}

### 3. Try a multiple linear regression

Next, you will fit a multiple linear regression to predict product sales, using all three features - TV ad budget, radio ad budget, and newspaper ad budget - to train *one* model.

The code to fit the model and look at the coefficients is provided for you.

:::

::: {.cell .markdown}
#### Fit a multiple linear regression
:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

reg_multi = LinearRegression().fit(train[['TV', 'radio', 'newspaper']], train['sales'])
```
:::


::: {.cell .markdown}
#### Look at coefficients
:::

::: {.cell .code}
```python
print("Coefficients (TV, radio, newspaper):", reg_multi.coef_)
print("Intercept: ", reg_multi.intercept_)
```
:::

::: {.cell .markdown }
#### Compute R2, MSE for multiple regression

Then, let's get:

* the predictions on the training set: `y_pred_tr_multi`
* the R2 score on the training set: `r2_tr_multi`
* the MSE on the training set: `mse_tr_multi`

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# y_pred_tr_multi = ...
# r2_tr_multi  = ...
# mse_tr_multi = ...
```
:::

::: {.cell .code}
```python
# print("Multiple regression R2:  ", r2_tr_multi)
# print("Multiple regression MSE: ", mse_tr_multi)

```
:::

::: {.cell .markdown}

#### Compute residuals

Compute the residuals ($y - \hat{y}$). 

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# residual_multi_tr = ...
```
:::


::: {.cell .markdown}

#### Plot predicted vs. actual sales

Next, plot predicted sales vs. actual sales for the multiple regression model.


* Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis). Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes.

:::


::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. What would you expect this plot to look like for a model that explains the data well? What does the plot tell you about your model in this specific case?

:::


::: {.cell .markdown}

#### Plot residuals vs. actual sales

Next, create a plot of residuals vs. actual sales for the multiple regression model.


* Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes, and each plot. 


:::

::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales?

:::


::: {.cell .markdown}

#### Plot residuals vs. features

Finally, you'll create a plot of residuals vs. features, for each of the three features, for your multiple regression model. Put your three subplots in one row.

In each subplot,

* create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis. 
* Use the same vertical scale for all subplots (but the horizontal scale will depend on the feature! The upper end of the TV ad spending range is different from the radio ad spending range.)  Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area for each subplot should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Make sure to clearly label each axis, and also label each subplot with a title that indicates which regression model it uses. 

:::

::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features?


:::

::: {.cell .markdown}

### 4. Decomposing the multiple regression with FWL

:::

::: {.cell .markdown}

In the previous section, you may have noticed that the coefficient for a given feature - say, newspaper - is different

* in the simple linear regression on newspaper
* and in the multiple regression which includes newspaper as well as other features.

In particular, in the simple linear regression we estimated that newspaper ad spending was associated with a positive effect on sales, similar in magnitude to TV ad spending. Now, the newspaper ads are estimated as having an association much closer to zero.

This is because:

* In the simple regression case, the coefficent for newspaper ads represents the effect of an increase in newspaper advertising.
* In the multiple regression case, the coefficient for newspaper ads represents the effect of an increase in newspaper advertising **while holding TV and radio advertising constant**.

It turns out that in the simple linear regression on newspaper ad budget, the regression was "learning" the effect of a feature that was correlated with newspaper ad budget, not the effect of newspaper ad budget itself. 

We observe that there is a correlation between newspaper ad budget and radio ad budget, and a smaller correlation between newspaper ad budget and TV ad budget. (This is logical; if an ad campaign spends a lot of money on one medium, they are likely to spend a lot on others as well.)

:::

::: {.cell .markdown}

You can see this pairwise correlation in the following table.

:::


::: {.cell .code}
```python
train[['TV', 'radio', 'newspaper']].corr()
```
:::


::: {.cell .markdown}

In the simple regression model, sales appear to increase when newspaper ad spending increases. However, this is because newspaper ad spending and other types of ad spending are correlated, so when newspaper ad spending increases the other types of ad spending also increase.

From the multiple regression model, we can see that when newspaper ad spending increases and other types of ad spending *do not* also increase, sales do not increase.

:::


::: {.cell .markdown}


In this section, we will explore this further using the Frisch-Waugh-Lovell (FWL) theorem. This theorem will help us understand in greater depth what the coefficient in the multiple regression represents, when there are correlated features included in the regression.

:::

::: {.cell .markdown}

#### Background: Frisch-Waugh-Lovell (FWL) theorem

Suppose we have a linear model with $k$ features:

$$\hat{y} = w_0 + w_1 x_1 + \ldots + w_k x_k + e$$

and residual $e = y - \hat{y}$, which is the part of $y$ that is not explained by the regression model.

The FWL theorem tells us that we can get $w_j$, the association between $x_j$ and $y$ *while holding other features constant*, with the following procedure to split up our multiple linear regression into *orthogonal* components - having no correlation.

:::


::: {.cell .markdown}

##### Step 1

First, we will split the feature $x_j$ into (1) the information that is already present in other features, and (2) the information that is not already in other features.

How do you split the feature $x_j$ into parts that are correlated and uncorrelated with other features? With another regression model! We will train a model using all features *except* $x_j$ to *predict* $x_j$, like this:

$$\hat{x_j} = w_0^{x_j} + \sum_{k \neq j} w^{x_{j}}_{k}x_{k}$$

where

* the superscript $x_j$ on the coefficients denotes that these are the coefficients for the model to predict $x_j$
* $\hat{x_j}$ is the estimate of $x_j$ according to this model, and this is the part of $x_j$ whose information is already in the other features. (This is the part that is correlated with other features.)
* the residual of this model, $\epsilon^{x_{j}} = x_j - \hat{x_j}$, is the part of $x_j$ whose information is *not* already in the other features. (This is the part that is uncorrelated with other features.)

:::

::: {.cell .markdown}

##### Step 2

Then, we'll split the target variable $y$ into (1) the part that can be predicted using the other features *without* $x_j$, and (2) the part that cannot be predicted without $x_j$. Of course, we will do this in a similar way, with another regression model:

$$\hat{y} = w_0^{y} + \sum_{k \neq j} w^{y}_{k}x_{k}$$

where

* the superscript $y$ on the coefficients denotes that these are the coefficients for the model to predict $y$, but without $x_j$
* the residual of this model, $\epsilon^{y} = y - \hat{y}$, is the part of $y$ that cannot be predicted without $x_j$.

:::

::: {.cell .markdown}

##### Step 3

This is the really interesting part! The FWL tells us that we can now train a model to:

* predict $\epsilon^y$, the part of $y$ that is *not* predicted by the regression on features excluding $x_j$
* as a simple regression on feature $\epsilon^{x_j}$, the part of $x_j$ that is *not* already "in" the other features

$$\epsilon^{y} = {w}^{*}_{j}\epsilon^{x_j} + \epsilon^{*}$$


and that

* the coefficient $w^{*}_{j}$ - which tells us the association between "the parts of $x_j$ not in the other features" and "the parts of $y$ not predicted by the other features" - is going to be the **same** as $w_j$ in the multiple regression model we fitted earlier!
* and the residuals $\epsilon^{*}$ will be the same as the residuals of the multiple regression model we fitted earlier.

This makes it clear that in a multiple regression model, the coefficient of the feature $x_j$ - which tells us the association between $x_j$ and $y$ when other features are held constant - tells us about how much of the variance in $y$ is explained by $x_j$ *independent of the other features in the model*.

:::

::: {.cell .markdown}

#### Apply FWL to the "newspaper" feature

Let us try it now on the `newspaper` feature. We noticed earlier that `newspaper` had a large positive coefficient in the simple regression model, but not in the multiple regression model. We also noticed that newspaper ad spending was strongly correlated with radio ad spending.

Note: we will use the training data *only* throughout this section.

:::

::: {.cell .markdown}

##### Step 1

First, use the *other* features (`TV` and `radio`) to train a model `reg_news_x` to "predict" `newspaper`.

* The prediction of the model tells us: how much of the "signal" in the `newspaper` feature is already present in the other features?
* The residual of the model tells us: how much of the "signal" in the `newspaper` feature is _not_ present in the other features? The "model" cannot predict this part of `newspaper` using the `TV` and `radio` features.


Save the model prediction on the training data in `xhat_news_x` and the residual for the training data in `residual_news_x`.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# reg_news_x  ...
# xhat_news_x = ...
# residual_news_x = ...
```
:::

::: {.cell .markdown}

We can see that the residual is the part of `newspaper` that is uncorrelated with `radio` and `TV`.

:::

::: {.cell .code}
```python
pd.concat([train['TV'], train['radio'], pd.Series(residual_news_x, name='residual')], axis=1).corr()
```
:::


::: {.cell .markdown}

while the prediction of that "model" is, of course, correlated with the features used for prediction - it is a linear combination of those features. (The pairwise correlation is not ideal for measuring this, since $w_j x_j$ appears as noise with respect to an uncorrelated feature $x_k$, but we can get some idea.)

:::

::: {.cell .code}
```python
pd.concat([train['TV'], train['radio'], pd.Series(xhat_news_x, name='prediction')], axis=1).corr()
```
:::

::: {.cell .markdown}

You can think of these two "components" of the "newspaper" feature as:

* the part of the "newspaper" feature that reflects spending specifically on the newspaper medium. (the residual)
* and the part that reflects the "spendiness" of the ad campaign, which is also reflected in TV and radio. (the prediction)

:::


::: {.cell .markdown}

Let's visualize these relationships. In the next cell, we will create a 2x2 grid of subplots:

* On the top row, we use seaborn's `regplot` to put `xhat_news_x` feature on the vertical axis, and the `TV` and `radio` features, respectively, on the horizontal axis. The `regplot` is a scatter plot with a simple linear regression line overlaid on top, to make it easier to see an association. Note that the horizontal scale of each subplot will be different, since the range of spending on TV ads is not the same as the range of spending on radio ads.
* On the bottom row, we will do the same, but with `residual_news_x` on the vertical axis.

:::

::: {.cell .code}
```python
plt.figure(figsize=(6,6))

plt.subplot(2,2,1)
sns.regplot(x=train['TV'], y=xhat_news_x);
plt.xlabel('TV ad $')
plt.ylabel('$\hat{x_j}$');


plt.subplot(2,2,2)
sns.regplot(x=train['radio'], y=xhat_news_x);
plt.xlabel('Radio ad $')
plt.ylabel('$\hat{x_j}$');

plt.subplot(2,2,3)
sns.regplot(x=train['TV'], y=residual_news_x);
plt.xlabel('TV ad $')
plt.ylabel('$\epsilon^{x_j}$');

plt.subplot(2,2,4)
sns.regplot(x=train['radio'], y=residual_news_x);
plt.xlabel('Radio ad $')
plt.ylabel('$\epsilon^{x_j}$');

plt.tight_layout();
```
:::


::: {.cell .markdown}

**Comment on the results**. 

- What do you notice about the relationship between $\hat{x_{j}}$ (`newspaper` ad spending) and the other features? To what extent can `newspaper` ad spending be "predicted" by the other features?
- You should observe that there is no relationship between $\epsilon^{x_{j}}$ (the residual of the model that is "trained to predict `newspaper`") and the other features. This residual represents what "part" of `newspaper` ad spending?

:::

::: {.cell .markdown}

##### Step 2

Now, train a model `reg_news_y` to predict "sales" *without* `newspaper`, using only `TV` and `radio` features. Save the prediction of this model on the training data in `yhat_news_y`, and the residuals of this model for the training data in `residual_news_y`.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# reg_news_y  = ...
# yhat_news_y = ...
# residual_news_y = ...
```
:::

::: {.cell .markdown}

and, we'll use this to generate a similar plot -

:::


::: {.cell .code}
```python
plt.figure(figsize=(6,6))

plt.subplot(2,2,1)
sns.regplot(x=train['TV'], y=yhat_news_y);
plt.xlabel('TV ad $')
plt.ylabel('$\hat{y}$');


plt.subplot(2,2,2)
sns.regplot(x=train['radio'], y=yhat_news_y);
plt.xlabel('Radio ad $')
plt.ylabel('$\hat{y}$');

plt.subplot(2,2,3)
sns.regplot(x=train['TV'], y=residual_news_y);
plt.xlabel('TV ad $')
plt.ylabel('$\epsilon^{y}$');

plt.subplot(2,2,4)
sns.regplot(x=train['radio'], y=residual_news_y);
plt.xlabel('Radio ad $')
plt.ylabel('$\epsilon^{y}$');

plt.tight_layout();
```
:::


::: {.cell .markdown}

Once again, you can think of these two "components" of `sales` as:

* the part of `sales` that is not predicted by the model on `TV` and `radio`. (the residual)
* and the part that *is* predicted by the model on `TV` and `radio`. (the prediction)

:::

::: {.cell .markdown}

**Comment on the results**. 

- What do you notice about the relationship between $\hat{y}$ (`sales`) and the other (not-$x_j$) features (`tv` ad spending, `radio` ad spending)?
- The residual $\epsilon^{y}$ represents the part of $\hat{y}$ (`sales`) that is still "unexplained" even after considering the linear effect of which features?


:::

::: {.cell .markdown}

##### Step 3

Finally, train a model `reg_news_fwl` on the residuals of the other two models! In this linear regression,

* the feature is `residual_news_x`
* the target variable is `residual_news_y`
* you will set `fit_intercept = False` - there is no $w_0$, since both feature and target variable are already zero-mean.

Save the predictions on the training set in `yhat_news_fwl`, and the residual in `residual_news_fwl`.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# reg_news_fwl  = ...
# yhat_news_fwl = ...
# residual_news_fwl = ...
```
:::

::: {.cell .markdown}

Check the coefficient of this model -

:::

::: {.cell .code}
```python
reg_news_fwl.coef_
```
:::



::: {.cell .markdown}

and compare it to the coefficient for `newspaper` in the simple regression `reg_news`, and in the multiple regression `reg_multi`.

:::

::: {.cell .markdown}

You can see that

* the association between "the part of `newspaper` *not* in `radio` or `TV`" and "the part of `sales` *not* predicted by `radio` or `TV`" (from this "residualized" model)
* is exactly the same as "the association between `newspaper` and `sales` when `radio` and `TV` are held constant" (from the multiple regression model fitted in the previous section).

Furthermore, this "residualized" model helps us understand that part of `residual_news_y` is "explained" by "the part of `newspaper` *not* in `radio` or `TV`", and part is not.

We can see this more clearly in the following plot, which shows:

* `sales` vs `newspaper` with the regression line showing the effect of `newspaper`, according to the regression on `newspaper` alone.
* `sales` vs `newspaper` with the regression line showing the effect of `newspaper`, according to the multiple regression model.
* and the "residualized" `sales` vs "residualized" `newspaper` showing the effect of `newspaper` according to the FWL regression.

:::


::: {.cell .code}
```python

plt.figure(figsize=(12, 4));

plt.subplot(1, 3, 1);
sns.scatterplot(data=train, x="newspaper", y="sales");
sns.lineplot(data=train, x="newspaper", y=reg_news.predict(train[['newspaper']]), color='red');
plt.ylim(0, 30);
plt.xlim(-20, 120);
plt.title("Simple Regression on Newspaper");

plt.subplot(1, 3, 2);
sns.scatterplot(data=train, x="newspaper", y="sales");
sns.lineplot(data=train, x="newspaper", y=train['sales'].mean() - (reg_multi.coef_[2] * train['newspaper'].mean()) + reg_multi.coef_[2] * train['newspaper'], color='red');
plt.ylim(0, 30);
plt.xlim(-20, 120);
plt.title("Multiple Regression");

plt.subplot(1, 3, 3);
sns.scatterplot(x=residual_news_x, y=residual_news_y);
sns.lineplot(x=residual_news_x, y=reg_news_fwl.predict(pd.DataFrame(residual_news_x)), color='red');
plt.title("FWL Regression on Residuals");
plt.ylim(-10, 10);
plt.xlim(-50, 90);

plt.tight_layout();
plt.show();
```
:::

::: {.cell .markdown}

**Comment on the results**. 

- What was the coefficient for `newspaper` in the simple regression `reg_news`, in the multiple regression `reg_multi`, and in the "residualized" model?
- What do the first two panels show about the association between the `newspaper` feature and `sales`?
- The third panel shows the residualized `sales` (after accounting for the effect of `tv` and `radio`) and the residualized `newspaper` (after removing the parts that are "predicted" by `tv` and `radio`). What does this plot show us about the association between `newspaper` and `sales`, once we remove the effect of `tv` and `radio` from both `newspaper` and `sales`?


:::


::: {.cell .markdown}

### 5. Linear regression with interaction terms

Finally, we will try to improve our regression by addressing patterns noted in the residuals of the multiple regression model.

Our multiple linear regression includes additive effects of all three types of advertising media. However, it does not include *interaction* effects, in which combining different types of advertising media together results in a bigger boost in sales than just the additive effect of the individual media.  

:::


::: {.cell .markdown}

#### Add interaction terms to the data

The pattern in the residuals plots from parts (1) through (3) suggest that a model including an interaction effect may explain sales data better than a model including additive effects. Add four columns to each data frame (`train` and `test`): 

* `newspaper` $\times$ `radio` (name this column `newspaper_radio`)
* `TV` $\times$ `radio` (name this column `TV_radio`)
* `newspaper` $\times$ `TV` (name this column `newspaper_TV`)
* `newspaper` $\times$ `radio` $\times$ `TV`  (name this column `newspaper_radio_TV`)

Note: you can use the `assign` function in `pandas` ([documentation here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html)) to create a new column and assign a value to it using operations on other columns.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

```
:::

::: {.cell .code}
```python
train.info()
```
:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

```
:::

::: {.cell .code}
```python
test.info()
```
:::

::: {.cell .markdown}

#### Fit a multiple linear regression with interaction terms

Then, train a linear regression model on all seven features: the three types of ad budgets, and the four interaction effects.

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# reg_inter = ....
```
:::


::: {.cell .markdown }
#### Compute R2, MSE for multiple regression with interaction terms

Then, let's get:

* the predictions on the training set: `y_pred_tr_inter`
* the R2 score on the training set: `r2_tr_inter`
* the MSE on the training set: `mse_tr_inter`

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# y_pred_tr_inter = ...
# r2_tr_inter  = ...
# mse_tr_inter = ...
```
:::

::: {.cell .code}
```python
# print("Multiple regression with interaction R2:  ", r2_tr_inter)
# print("Multiple regression with interaction MSE: ", mse_tr_inter)

```
:::

::: {.cell .markdown}

#### Compute residuals

Compute the residuals ($y - \hat{y}$). 

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# residual_inter_tr = ...
```
:::


::: {.cell .markdown}

#### Plot predicted vs. actual sales

Next, plot predicted sales vs. actual sales for the multiple regression model with interaction terms.


* Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis). Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes.

:::


::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. What would you expect this plot to look like for a model that explains the data well? What does the plot tell you about your model in this specific case? In particular, compare what you observe here to your observations from the similar plot on the multiple regression model without interaction terms.

:::


::: {.cell .markdown}

#### Plot residuals vs. actual sales

Next, create a plot of residuals vs. actual sales for the multiple regression model with interaction terms.


* Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. 
* Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Label each axes, and each plot. 


:::

::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales? In particular, compare what you observe here to your observations from the similar plot on the multiple regression model without interaction terms.

:::


::: {.cell .markdown}

#### Plot residuals vs. features

Finally, you'll create a plot of residuals vs. features, for each of the three **original** features (not the interaction terms), for your multiple regression model with interaction terms. Put your three subplots in one row.

In each subplot,

* create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis. 
* Use the same vertical scale for all subplots (but the horizontal scale will depend on the feature! The upper end of the TV ad spending range is different from the radio ad spending range.) Make sure the scale is appropriate (does not exclude data, and does not make it difficult to see the data due to excessive whitespace). Also, the plot area for each subplot should be square-shaped (similar height and width) in order to make the relevant trend easier to see.
* Make sure to clearly label each axis, and also label each subplot with a title that indicates which regression model it uses. 

:::

::: {.cell .code}
```python
# plot

```
:::

::: {.cell .markdown}

**Comment on this plot**. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features? In particular, compare what you observe here to your observations from the similar plot on the multiple regression model without interaction terms.


:::


::: {.cell .markdown }
#### Compute test R2, MSE for multiple regression with interaction terms

Finally, use this fitted model to get:

* the predictions on the **test** set: `y_pred_ts_inter`
* the R2 score on the **test** set: `r2_ts_inter`
* the MSE on the **test* set: `mse_ts_inter`

:::

::: {.cell .code}
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# y_pred_ts_inter = ...
# r2_ts_inter  = ...
# mse_ts_inter = ...
```
:::

::: {.cell .code}
```python
# print("Multiple regression with interaction R2 - test set:  ", r2_ts_inter)
# print("Multiple regression with interaction MSE- test set: ", mse_ts_inter)

```
:::

