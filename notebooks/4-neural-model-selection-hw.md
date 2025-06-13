---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: 'text/x-python'
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.7
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Model Order Selection for Neural Data
:::

::: {.cell .markdown }
**Name**: \[Fill in your name here\]
**Net ID**: \[Fill in your net ID here\]

:::


::: {.cell .markdown}


Make a copy of this notebook in your own Google Drive and, as you work through it, fill in missing code and answers to the questions.

After you are finished, you will copy your answers from individual sections *and* a copy of the entire notebook into PrairieLearn for submission. (Note that the PrairieLearn autograder will expect that you have used exactly the variable names shown in this template notebook.)

Answers to open-ended questions (e.g. "Comment on the results..") must be **in your own words**, reflecting your own interpretation and understanding, and must refer to specific results (including numeric values) you obtained in this notebook.

:::


::: {.cell .markdown }
**Attribution**: This notebook is a slightly adapted version of the
[model order selection lab
assignment](https://github.com/sdrangan/introml/blob/master/unit04_model_sel/lab_neural_partial.ipynb)
by Prof. Sundeep Rangan.
:::

::: {.cell .markdown }
Machine learning is a key tool for neuroscientists to understand how
sensory and motor signals are encoded in the brain. In addition to
improving our scientific understanding of neural phenomena,
understanding neural encoding is critical for brain machine interfaces.
In this notebook, you will use model selection for performing some
simple analysis on real neural signals.
:::

::: {.cell .markdown }
## Loading the data

The data in this lab comes from neural recordings described in:

`<a href="http://jn.physiology.org/content/106/2/764.short">`{=html}
Stevenson, Ian H., et al. "Statistical assessment of the stability of
neural movement representations." Journal of neurophysiology 106.2
(2011): 764-774`</a>`{=html}

Neurons are the basic information processing units in the brain. Neurons
communicate with one another via *spikes* or *action potentials* which
are brief events where voltage in the neuron rapidly rises then falls.
These spikes trigger the electro-chemical signals between one neuron and
another. In this experiment, the spikes were recorded from 196 neurons
in the primary motor cortex (M1) of a monkey using an electrode array
implanted onto the surface of a monkey\'s brain. During the recording,
the monkey performed several reaching tasks and the position and
velocity of the hand was recorded as well.

The goal of the experiment is to try to *read the monkey\'s brain*: That
is, predict the hand motion from the neural signals from the motor
cortex.

We first load the key packages.
:::

::: {.cell .code }
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
```
:::

::: {.cell .markdown }
The full data is available on the CRCNS website
<http://crcns.org/data-sets/movements/dream>. However, the raw data
files can be quite large. To make the lab easier, the [Kording
lab](http://kordinglab.com/) at UPenn has put together an excellent
[repository](https://github.com/KordingLab/Neural_Decoding) where they
have created simple pre-processed versions of the data. You can download
the file `example_data_s1.pickle` from the [Dropbox
link](https://www.dropbox.com/sh/n4924ipcfjqc0t6/AADOv9JYMUBK1tlg9P71gSSra/example_data_s1.pickle?dl=0).
Alternatively, you can directly run the following command. This may take
a little while to download since the file is 26 MB.
:::

::: {.cell .code }
```python
!wget 'https://www.dropbox.com/sh/n4924ipcfjqc0t6/AADOv9JYMUBK1tlg9P71gSSra/example_data_s1.pickle?dl=1' -O example_data_s1.pickle
```
:::

::: {.cell .markdown }
The file is a *pickle* data structure, which uses the Python package
`pickle` to serialize Python objects into data files. Once you have
downloaded the file, you can run the following command to retrieve the
data from the pickle file.
:::

::: {.cell .code }
```python
with open('example_data_s1.pickle', 'rb') as fp:
    X,y = pickle.load(fp)
```
:::

::: {.cell .markdown }
The matrix `X` is matrix of spike counts from different neurons, where
`X[i,j]` is the number of spikes from neuron `j` in time bin `i`.

The matrix `y` has two columns:

-   `y[i,0] =` velocity of the monkey\'s hand in the x-direction in time
    bin `i`
-   `y[i,1] =` velocity of the monkey\'s hand in the y-direction in time
    bin `i`

Our goal will be to predict `y` from `X`.

Each time bin represent `tsamp=0.05` seconds of time. Using `X.shape`
and `y.shape`, we can compute and print:

-   `nt =` the total number of time bins
-   `nneuron =` the total number of neurons
-   `nout =` the total number of output variables to track = number of
    columns in `y`
-   `ttotal =` total time of the experiment is seconds.
:::

::: {.cell .code }
```python
tsamp = 0.05  # sampling time in seconds

nt, nneuron = X.shape
nout = y.shape[1]
ttotal = nt*tsamp

print('Number of neurons = %d' % nneuron)
print('Number of time samples = %d' % nt)
print('Number of outputs = %d' % nout)
print('Total time (secs) = %f' % ttotal)
```
:::

::: {.cell .markdown }
Then, we can plot the velocity against time, for each direction, for the
first 1000 samples:
:::

::: {.cell .code }
```python
t_cutoff = 1000
directions = ['x', 'y']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,7))
for n in range(nout):
  sns.lineplot(x=np.arange(0, t_cutoff)*tsamp, y=y[0:t_cutoff, n], label=directions[n], ax=axes[n]);

  axes[n].set_ylabel("Velocity")
  axes[n].set_xlabel("Time (s)")
  axes[n].set_ylim(-50,50)
```
:::

::: {.cell .markdown }
We can also "zoom in" on a small slice of time in which the monkey is
moving the hand, and see the neural activity at the same time.
:::

::: {.cell .code }
```python
t_start = 490
t_end = 580

fig, axes = plt.subplots(nrows=7, ncols=8, figsize=(15,15))

# Setting the range for all axes
plt.setp(axes, xlim=(t_start, t_end), ylim=(0,12));

for n in range(nout):
  sns.lineplot(x=np.arange(t_start, t_end)*tsamp, y=y[t_start:t_end, n], ax=axes[n//2,n%2], color='red', label=directions[n])
  plt.setp(axes[n//2,n%2], xlim=(t_start*tsamp, t_end*tsamp), ylim=(-50, +50));

for n in range(nneuron):
  sns.lineplot(x=np.arange(t_start, t_end)*tsamp, y=X[t_start:t_end, n], ax=axes[(n+2)//8,(n+2)%8], label="n%d" % n, color='grey')
  plt.setp(axes[(n+2)//8,(n+2)%8], xlim=(t_start*tsamp, t_end*tsamp), ylim=(0, +15));

```
:::

::: {.cell .markdown }
## 1. Fitting a linear model

Let's first try a linear regression model to fit the data.

To start, we will split the data into a training set and a test set.
We'l fit the model on the training set and then use the test set to
estimate the model performance on new, unseen data.
:::

::: {.cell .markdown }
**To shuffle or not to shuffle?**

The `train_test_split` function has an optional `shuffle` argument.

-   If you use `shuffle=False`, then `train_test_split` will take the
    first part of the data as the training set and the second part of
    the data as the test set, according to the ratio you specify in
    `test_size` or `train_size`.
-   If you use `shuffle=True`, then `train_test_split` will first
    randomly shuffle the data. Then, it will take the first part of the
    *shuffled* data as the training set and the second part of the
    *shuffled* data as the test set, according to the ratio you specify
    in `test_size` or `train_size`.

According to the function
[documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
by default, `shuffle` is `True`:

> **shuffle: bool, default=True**
>
> Whether or not to shuffle the data before splitting. If shuffle=False
> then stratify must be None.

so if you do not specify anything related to `shuffle`, your data will
be randomly shuffled before it is split into training and test data.

Under what conditions should you shuffle data? Suppose your dataset
includes samples of a medical experiment on 1000 subjects, and the first
500 samples in the data are from male subjects while the second 500
samples are from female subjects. If you set `shuffle=False`, then your
training set would have a much higher proportion of male subjects than
your test set (with the specific numbers depending on the ratio you
specify).

On the other hand, suppose your dataset includes stock prices at closing
time, with each sample representing a different date (in order). If you
allow `train_test_split` to shuffle the data, then your model will be
allowed to "learn" stock prices using prices from the day *after* the
one it is trying to predict! Obviously, your model won't be able to
learn from future dates in production, so it shouldn't be allowed to in
the evaluation stage, either. (Predicting the past using the future is
considered a type of data leakage.)

With this in mind, it is usually inappropriate to shuffle time series
data when splitting it up into smaller sets for training, validation, or
testing.

(There are more sophisticated ways to handle splitting time series data,
but for now, splitting it up the usual way, just without shuffling
first, will suffice.)
:::

::: {.cell .markdown }
Given the discussion above, use the `train_test_split` function to split
the data into training and test sets, but with no shuffling. Let
`Xtr,ytr` be the training data set and `Xts,yts` be the test data set.
Use `test_size=0.33` so 1/3 of the data is used for evaluating the model
performance.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Split data intro training and test sets
# Xtr ...
```
:::

::: {.cell .markdown }
Now, fit a linear regression on the training data `Xtr,ytr`. Make a
prediction `yhat_no_dly` using the test data, `Xts`. Compare `yhat_no_dly` to `yts` to
measure `rsq_no_dly`, the R2 value. Use the sklearn `r2_score` method.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Fit a linear model
# yhat_no_dly = ...
# rsq_no_dly = ...
```
:::


::: {.cell .markdown}
Print the `rsq_no_dly` value. You should get `rsq_no_dly` of around `0.45`.
:::


::: {.cell .code}
```python
rsq_no_dly
```
:::


::: {.cell .markdown }
It is useful to plot the predicted vs. actual values. 
Use the test data for this visualization.
Create a scatter plot of predicted values ($\hat{y}$) on the
vertical axis, and actual values ($y$) on the horizontal axis. 
Since we have two predicted values for each sample - 
the velocity in the X direction and the velocity in the Y direction - 
you should make two subplots,

-   one of predicted X direction vs. actual X direction,
-   one of predicted Y direction vs. actual Y direction

Make sure both axes use the same scale (the range of the vertical axis should be
the same as the range of the horizontal axis) *and* that all 
subplots use the same scale. Label each axes, and each plot 
(indicate which plot shows the velocity in the X direction and 
which shows the velocity in the Y direction!)  Also, the plot area for 
each subplot should be square-shaped (similar height and width) in order 
to make the relevant trend easier to see.

:::

::: {.cell .code }
```python
# TODO: Predicted values vs true values visualization
```
:::

::: {.cell .markdown }
It can also be useful to visualize the actual and predicted values over
time, for a slice of time. Use the test data for this visualization.
Create two subplots, both with time on the
horizontal axis, but only including *the first 1000 rows* (50 seconds) in the data.
On the vertical axis,

-   for one subplot: show the actual X direction as a line of one color,
    and the predicted X direction as a line of another color.
-   for the second subplot: show the actual Y direction as a line of one
    color, and the predicted Y direction as a line of another color.

Make sure to carefully label each axis (including units on the time
axis!), and label the data series (i.e. which color is the actual value
and which is the predicted value). Use solid lines (not dashed or dotted).
Use similar dimensions as the "velocity vs. time" plot above.

:::

::: {.cell .code }
```python
# TODO: Predicted and true values over time visualization
```
:::

::: {.cell .markdown }
Comment on this plot - does the model predict the hand velocity well?
:::

::: {.cell .markdown }
## 2. Fitting a model with delay
:::

::: {.cell .markdown }
One way we can improve the model accuracy is to add features using
delayed version of the existing features.

Specifically, the model we used above tries to predict velocity in
direction $k$ at time $i$ using

$$\hat{y}_{i,k} = w_{k,0} + \sum_{d=1}^{\text{nneuron}} w_{k,d} X_{i,d}  $$

In this model, $\hat{y}_{i,k}$ at the $i$th time bin was only dependent
on $X_i$, the number of spikes of each neuron in time bin $i$. In signal
processing, this is called a *memoryless* model.
:::

::: {.cell .markdown }
However, in many physical systems, such as those that arise in
neuroscience, there is a delay between the inputs and outputs. To model
this effect, we could add additional features to each row of data,
representing the number of spikes of each neuron in the *previous* row.
Then, the output at time $i$ would be modeled as the effect of the
neurons firing in time $i$ *and* the effect of the neurons firing in
time $i-1$.

We wouldn't be able to use data from the past for the first row of
data, since we don't *have* data about neurons firing in the previous
time step. But we can drop that row. If our original data matrix had
`nt` rows and `nneuron` columns, our data matrix with delayed features
would have `nt - 1` rows and `nneuron + 1 x nneuron` columns. (The first
`nneuron` columns represent the number of spikes in each neuron for the
current time, the next `nneuron` columns represent the number of spikes
in each neuron for the previous time.)

Furthermore, we can "look back" any number of time steps, so that the
output at time $i$ is modeled as the effect of the neurons firing in
time $i$, the neurons firing in time $i-1$, \..., all the way up to the
effect of the neurons firing in time $i- \text{dly}$ (where $\text{dly}$
is the maximum number of time steps we\'re going to "look back" on).
Our data matrix with the additional delayed features would have
`nt - dly` rows and `nneuron + dly x nneuron` columns.
:::

::: {.cell .markdown }
Here is a function that accepts `X` and `y` data and a `dly` argument,
and returns `X` and `y` with delayed features up to `dly` time steps
backward.
:::

::: {.cell .code }
```python
def create_dly_data(X,y,dly):
    """
    Create delayed data
    """    
    n,p = X.shape
    Xdly = np.zeros((n-dly,(dly+1)*p))
    for i in range(dly+1):
        Xdly[:,i*p:(i+1)*p] = X[dly-i:n-i,:]
    ydly = y[dly:]
    
    return Xdly, ydly
```
:::

::: {.cell .markdown }
To convince yourself that this works, try creating a data matrix that
includes delayed features one time step back:
:::

::: {.cell .code }
```python
X_dly_1, y_dly_1 = create_dly_data(X, y, 1)
```
:::

::: {.cell .markdown }
Verify that the dimensions have changed, as expected:
:::

::: {.cell .code }
```python
# dimensions of original data matrix
X.shape
```
:::

::: {.cell .code }
```python
# dimensions of data matrix with delayed features 1 time step back
X_dly_1.shape
```
:::

::: {.cell .markdown }
Check row 0 in the matrix with delayed features, and verify that it is
the concatenation of row 1 and row 0 in the original data matrix. (Note
that row 0 in the matrix with delayed features corresponds to row 1 in
the original data matrix.)
:::

::: {.cell .code }
```python
X_dly_1[0]
```
:::

::: {.cell .code }
```python
np.hstack((X[1], X[0]))
```
:::

::: {.cell .code }
```python
y_dly_1[0]
```
:::

::: {.cell .code }
```python
y[1]
```
:::

::: {.cell .markdown }
Now fit an linear delayed model with `dly=2` delay lags. That is,

-   Create delayed data `X_dly_2, y_dly_2` by calling `create_dly_data` with `dly=2`
-   Split the data (with the extra delay features!) into training and test as before (again, do not
    shuffle the data, and use a test size of 0.33). Name the training data `Xtr_dly_2, ytr_dly_2` and name the test data `Xts_dly_2, yts_dly_2`.
-   Fit the model on the training data
-   Use the model to predict the values for the test data and save the result in `yhat_dly_2`
-   Measure the R2 score on the test data and save the result in `rsq_dly_2`. (Note: make sure you are comparing the model prediction to the "delayed" test data samples, not the original samples!)

If you did this correctly, you should get a new R2 score around 0.60.
This is significantly better than the memoryless model.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Fit a linear model with dly=2

# Create the delayed data
# X_dly_2, y_dly_2 = ...

# Split into training and test
# Xtr_dly_2, ytr_dly_2 = ...
# Xts_dly_2, yts_dly_2 = ...

# Create linear regression object

# Fit the model

# Predict values for test data
# yhat_dly_2 = ... 

# Measure the new r2 score
# rsq_dly_2 = ...
```
:::

::: {.cell .code }
```python
rsq_dly_2
```
:::


::: {.cell .markdown }
As before (with the same style requirements as before), plot the predicted vs. true values, but for the model 
with `dly=2`, with one subplot for X velocity and one subplot for 
Y velocity.
:::

::: {.cell .code }
```python
# TODO: Predicted values vs true values visualization with dly=2

```
:::

::: {.cell .markdown }
Also as you did before (with the same style requirements as before), plot the actual and predicted values over time 
for the first 1000 samples, for the model with `dly=2`. (Note: make sure you are comparing the model prediction to the "delayed" test data samples, not the original samples!)


Does the model predict the hand velocity well?
:::

::: {.cell .code }
```python
# TODO: Predicted and true values over time visualization with dly=2
```
:::

::: {.cell .markdown }
## 3. Selecting the optimal delay with K-fold CV

In the previous example, we fixed `dly=2`. We will now select the optimal
delay using K-fold cross validation.

Since we have a large number of data samples, it will take a long time to run the K-fold CV
for finding the optimal delay. So, to simplify things, we will
just pretend that we have a very limited data set, `Xred` and `yred`.
In this section, we'll use the reduced data set `Xred` and `yred` in place
of the original `X` and `y`.

We will compute `Xred` and `yred` by taking the first `nred=6000`
samples of the data `X` and `y`.
:::

::: {.cell .code }
```python
nred = 6000

Xred = X[:nred]
yred = y[:nred]
```
:::

::: {.cell .markdown }

Note: since we are only using the first 6000 samples to train the model
and select the best `dly`, there are plenty of samples left out as the test set.
We don't need to (and shouldn't) further divide these 6000 samples into training
and test sets - we can use all of it for model training and model selection.
:::

::: {.cell .markdown }
We will look at model orders up to `dmax=15`. We will create a delayed
data matrix, `X_dly_15,y_dly_15`, using `create_dly_data` with the reduced data
`Xred,yred` and `dly=dmax`.
:::

::: {.cell .code }
```python
dmax = 15
X_dly_15, y_dly_15 = create_dly_data(Xred,yred,dmax)
```
:::

::: {.cell .code }
```python
X_dly_15.shape
```
:::

::: {.cell .code }
```python
y_dly_15.shape
```
:::


::: {.cell .markdown }
We are going to use K-fold CV with `nfold=10` to find the optimal delay,
for all the values of delay in `dtest_list`:
:::

::: {.cell .code }
```python
dtest_list = np.arange(0, dmax+1)
nd = len(dtest_list)

print(dtest_list)
```
:::

::: {.cell .markdown }
You can refer to the example in the "Model order selection" section of
the demo notebook. But, make sure to use `shuffle=False` in your `KFold`
object, since for this example it would be inappropriate to shuffle the
data.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)


# TODO: Use K-fold CV to select dly

# Number of folds
nfold = 10
 
#  Create a k-fold object
# kf = KFold(...)
 
# Initialize a matrix rsq_kf to hold values of the R^2 across the model orders and folds.
rsq_kf = np.zeros((nd,nfold))
 
# Loop over the folds
for i, idx_split in enumerate(kf.split(Xdly)):
    
    # Get the training and validation data in the split
    idx_tr, idx_val = idx_split        
 
    for it, dtest in enumerate(dtest_list):
        # DO NOT call create_dly_data again
        # just select the appropriate subset of columns of X_dly_15
        # X_dtest = X_dly_15 with the columns corresponding to only the `dtest+1` most recent times.
 
        # Split the data (X_dtest, y_dly_15) into training and validation
        # using idx_tr and idx_val
        # Xtr_kf = ...
        # ytr_kf = ...
        # Xval_kf = ...
        # yval_kf = ...
 
        # Fit linear regression on training data
 
        #  Measure the R2 on validation data and store in the matrix rsq_kf
        # yhat_kf = ...
```
:::

::: {.cell .markdown }
Write code to find the delay that has the best mean validation R2. Get the
best delay according to the "best R2" rule, and save it in `d_opt`. Print the value of `d_opt`.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Use K-fold CV (continued)
# d_opt = ....
```
:::



::: {.cell .markdown }
Now write code to find the best delay using the one SE rule (i.e. find
the simplest model whose validation R2 is within one SE of the model
with the best R2). 

* get the "target R2" that is within one of the "best R2" model, and save it in `rsq_one_se_tgt`
* then get the delay of the simplest model with R2 greater than this target, and save it in `d_one_se`.

Print the value of `d_one_se`.
:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Use K-fold CV (continued)
```
:::


::: {.cell .markdown }
Plot the mean and standard error of the validation R2 values for each model
(each delay value) as a function of the delay. Use a
`plt.errorbar` plot, as shown in the "Model selection using 1-SE rule"
section of the demo notebook. Label each axes.
:::

::: {.cell .code }
```python
# TODO: Visualize mean and SE of R2 across folds for each dly
```
:::

::: {.cell .markdown }

## 4. Fitting the selected model
:::

::: {.cell .markdown }
Now that we have selected a model order, we can fit a 
linear regression model using the best delay according to the one SE rule.

First, create an "best according to one-SE delay" training and test set

:::


::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# Un-comment this line:
# X_dly_one_se, y_dly_one_se = create_dly_data(X, y, d_one_se)

# Then, use train_test_split to create Xtr_dly_one_se, ytr_dly_one_se and Xts_dly_one_se, yts_dly_one_se
# (using the same settings as before)
# Xtr_dly_one_se, ytr_dly_one_se = ...
# Xts_dly_one_se, yts_dly_one_se = ...
```
:::

::: {.cell .markdown }

Then,  fit a linear model:

:::


::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Fit a linear model with `dly=d_one_se`

```
:::


::: {.cell .markdown }
Use your fitted model to get the predictions `yhat_dly_one_se` 
and find the R2 score `rsq_dly_one_se` on the test set. 

Print the value of `rsq_dly_one_se`. It should be substantially
better than before.

:::

::: {.cell .code }
```python
#grade (write your code in this cell and DO NOT DELETE THIS LINE)

# TODO: Fit a linear model with `dly=d_one_se` (continued) 
# yhat_dly_one_se = ...
# rsq_dly_one_se = ...
```
:::



::: {.cell .markdown }
Also plot the actual and predicted values over time for the first 1000
samples of the *test* data (similar to your plots in the previous
sections, following the same style requirements). 
(Note: make sure you are comparing the model prediction to the "delayed" 
test data samples, not the original samples!)


Comment on this plot - does the model predict the hand
velocity well, compared to the previous models? 

See if you can identify a few points where this model 
does noticeably better. Download and *annotate* this plot - 
circle parts of the time series where the model with one-SE delay
is noticably better at predicting the true value than the model with delay = 2.

Comment on the versions of this plot you created for 
each of the three models (no delay, delay 2, delay with one-SE delay).
What is the R2 score of the model in each case? Does the visual improvement
you can see in the plot align with your expectations, based on the magnitude
of the improvement in R2 score?


:::

::: {.cell .code }
```python
# TODO: Predicted and true values over time for dly=d_one_se visualization
```
:::

-->
