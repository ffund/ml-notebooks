---
title: 'Voter classification using exit poll data'
author: 'Fraida Fund'
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 0
---


::: {.cell .markdown}
# Assignment: Voter classification using exit poll data

_Fraida Fund_

:::

::: {.cell .markdown}

**TODO**: Edit this cell to fill in your NYU Net ID and your name:

-   **Net ID**:
-   **Name**:

:::

::: {.cell .markdown}

In this notebook, we will explore the problem of voter classification.

Given demographic data about a voter and their opinions on certain key issues, can we predict their vote in the 2016 U.S. presidential election? We will attempt this using a K nearest neighbor classifier.

In the first few sections of this notebook, I will show you how to prepare the data and and use a K nearest neighbors classifier for this task, including:

* getting the data and loading it into the workspace.
* preparing the data: dealing with missing data, encoding categorical data in numeric format, and splitting into training and test.

In the last few sections of the notebook, you will have to improve the basic model for better performance, using a custom distance metric and using feature weighting. In these sections, you will have specific criteria to satisfy for each task. 

**However, you should also make sure your overall solution is good!** An excellent solution to this problem will achieve greater than 80% validation accuracy. A great solution will achieve 75% or higher.
:::


::: {.cell .markdown}

#### 📝 Specific requirements

* For full credit, you should achieve 75% or higher test accuracy overall in this notebook (i.e. when running your solution notebook from beginning to end).
* If your solution achieves an especially excellent accuracy result relative to your classmates', you may also earn extra credit toward this lab grade (at my discretion).

:::

::: {.cell .markdown}
## Import libraries
:::


::: {.cell .code}
```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.set_printoptions(suppress=True)
```
:::


::: {.cell .markdown}
## Load data
:::


::: {.cell .markdown}
The data for this notebook comes from the [U.S. National Election Day Exit Polls](https://ropercenter.cornell.edu/exit-polls/us-national-election-day-exit-polls).

Here's a brief description of how exit polls work. 
 
Exit polls are conducted by Edison Research on behalf of a consortium of media organizations.

First, the member organizations decide what races to cover, what sample size they want, what questions should be asks, and other details. Then, sample precincts are selected, and local interviewers are hired and trained. Then, at those precincts, the local interviewer approaches a subset of voters as they exit the polls (for example, every third voter, or every fifth voter, depending on the required sample size).
 
When a voter is approached, they are asked if they are willing to fill out a questionnaire. Typically about 40-50% agree. (For those that decline, the interviewer visually estimates their age, race, and gender, and notes this information, so that the response rate by demographic is known and responses can be weighted accordingly in order to be more representative of the population.)
 
Voters that agree to participate are then given an form with 15-20 questions. They fill in the form (anonymously), fold it, and put it in a small ballot box.
 
Three times during the day, the interviewers will stop, take the questionnaires, compile the results, and call them in to the Edison Research phone center.  The results are reported immediately to the media organizations that are consortium members.
 
In addition to the poll of in-person voters, absentee and early voters (who are not at the polls on Election Day) are surveyed by telephone.

:::


::: {.cell .markdown}

### Download the data and documentation

The exit poll data is not freely available on the web, but is available to those with institutional membership. You will be able to use your NYU email address to create an account with which you can download the exit poll data.

To get the data:

1. Visit [the Roper Center website via NYU Libraries link](https://persistent.library.nyu.edu/arch/NYU02495). Click on the user icon in the top right of the page, and choose "Log in".
2. For "Your Affiliation", choose "New York University".
3. Then, click on the small red text "Register" below the password input field. The email and password fields will be replaced by a new email field with two parts. 
4. Enter your NYU email address in the email field, and then click the red "Register" button.
5. You will get an email at your NYU email address with the subject "Roper iPoll Account Registration". Open the email and click "Confirm Account" to create a password and finish your account registration.
6. Once you have completed your account registration, log in to Roper iPoll by clicking the user icon in the top right of the page, choosing "Log in", and entering your NYU email address and password.
7. Then, open the Study Record for the [2016 National Election Day Exit Poll](https://ropercenter.cornell.edu/ipoll/study/31116396).
8. Click on the "Downloads" tab, and then click on the CSV data file in the "Datasets" section of this tab. Press "Accept" to accept the terms and conditions. Find the file `31116396_National2016.csv` in your browser's default download location.
9. After you download the CSV file, scroll down a bit until you see the "Study Documentation, Questionnaire and Codebooks" PDF file. Download this file as well.

:::


::: {.cell .markdown}

### Upload into Colab filesystem

To get the data into Colab, run the following cell. Upload the CSV file you just downloaded (`31116396_National2016.csv`) to your Colab workspace. Wait until the uploaded has **completely** finished - it may take a while, depending on the quality of your network connection.
:::


::: {.cell .code}
```python
try:
  from google.colab import files

  uploaded = files.upload()

  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))

except:
  pass # not running in Colab
```
:::


::: {.cell .markdown}

### Load data with pandas

Now, use the `read_csv` function in `pandas` to read in the file. 

Also use `head` to view the first few rows of data and make sure that everything is read in correctly.

:::



::: {.cell .code}
```python
df = pd.read_csv('31116396_National2016.csv')
df.head()
```
:::


::: {.cell .markdown}
## Prepare data
:::


::: {.cell .markdown}
Survey data can be tricky to work with, because surveys often "branch"; the questions that are asked depends on a respondent's answers to other questions.

In this case, different respondents fill out different versions of the survey. Review pages 7-11 of the "Study Documentation, Questionnaire, and Codebooks" PDF file you downloaded earlier, which shows the five different questionnaire versions used for the 2016 exit polls. 

Note that in a red box next to each question, you can see the name of the variable (column name) that the respondent's answer will be stored in.

:::


::: {.cell .markdown}
![Exit poll versions](https://raw.githubusercontent.com/ffund/ml-notebooks/master/notebooks/images/exit-poll-survey-versions.png)
:::


::: {.cell .markdown}
This cell will tell us how many respondents answered each version of the survey:
:::


::: {.cell .code}
```python
df['VERSION'].value_counts()
```
:::


::: {.cell .markdown}
Because each respondent answers different questions, for each row in the data, only some of the columns - the columns corresponding to questions included in that version of the survey - have data. Our classifier will need to handle that.
:::


::: {.cell .markdown}

You may also notice that the data is *categorical*, not *numeric* - for each question, users choose their response from a finite set of possible answers. We will need to convert this type of data into something that our classifier can work with.

:::




::: {.cell .markdown}
### Label missing data
:::


::: {.cell .markdown}
Since each respondent only saw a subset of questions, we expect to see missing values in each column.

However, if we look at the **count** of values in each column, we see that there are no missing values - every column has the full count!
:::


::: {.cell .code}
```python
df.describe(include='all')
```
:::

::: {.cell .markdown}
This is because missing values are recorded as a single space, and not with a NaN. 

Let's change that:
:::

::: {.cell .code}
```python
df.replace(" ", float("NaN"), inplace=True)
```
:::

::: {.cell .markdown}
Now we can see an accurate count of the number of responses in each column: 
:::

::: {.cell .code}
```python
df.describe(include='all')
```
:::


::: {.cell .markdown}
Notice that *every* row has some missing data! If we drop the rows with missing values, we're left with an empty data frame (0 rows):
:::


::: {.cell .code}
```python
df.dropna()
```
:::


::: {.cell .markdown}
Instead, we'll have to make sure that the classifier we use is able to work with partial data. One nice benefit of K nearest neighbors is that it can work well with data that has missing values, as long as we use a distance metric that behaves reasonably under these conditions.
:::


::: {.cell .markdown}
### Encode target variable as a binary variable
:::


::: {.cell .markdown}
Our goal is to classify voters based on their vote in the 2016 presidential election, i.e. the value of the `PRES` column. We will restrict our attention to the candidates from the two major parties, so we will throw out the rows representing voters who chose other candidates:
:::

::: {.cell .code}
```python
df = df[df['PRES'].isin(['Donald Trump', 'Hillary Clinton'])]
df.reset_index(inplace=True, drop=True)
df.info()
```
:::


::: {.cell .code}
```python
df['PRES'].value_counts()
```
:::


::: {.cell .markdown}
Now, we will transform the string value into a binary variable, and save the result in `y`. We will build a binary classifier that predicts `1` if it thinks a sample is  Trump voter, and `0` if it thinks a sample is a Clinton voter.
:::

::: {.cell .code}
```python
y = df['PRES'].map({'Donald Trump': 1, 'Hillary Clinton': 0}) 
y.value_counts()
```
:::



::: {.cell .markdown}
### Encode ordinal features
:::


::: {.cell .markdown}
Next, we need to encode our features. All of the features are represented as strings, but we will have to transform them into something over which we can compute a meaningful distance measure.

Columns that have a **logical order** should be encoded using ordinal encoding, so that the distance metric will be meaningful.

For example, consider the `AGE` column, in which users select an option from the following:
:::


::: {.cell .code}
```python
df['AGE'].unique()
```
:::


::: {.cell .markdown}
What if we transform the `AGE` column using four binary columns: `AGE_18-29`, `AGE_30-44`, `AGE_45-65`, `AGE_65+`, with a 0 or a 1 in each column to indicate the respondent's age?

If we did this, we would lose meaningful information about the distance between ages; a respondent whose age is 18-29 would have the same distance to one whose age is 45-65 as to one whose age is 65+. Logically, we expect that a respondent whose age is 18-29 is most similar to the other 18-29 respondents, less similar to the 30-44 respondents, even less similar to the 45-65 respondents, and least similar to the 65+ respondents. 

To realize this, we will use **ordinal encoding**, which will represent `AGE` in a single column with *ordered* integer values.

:::


::: {.cell .markdown}

First, we define a dictionary that maps each possible value to an integer. 

:::


::: {.cell .code}
```python
mapping_dict_age = {'18-29': 1, 
                 '30-44': 2,
                 '45-65': 3,
                 '65+': 4}
```
:::


::: {.cell .markdown}

Then we can create a new data frame, `df_enc_ord`, by calling `map` on the original `df['AGE']` and passing this mapping dictionary. We will also specify that the index should be the same as the original data frame:

:::


::: {.cell .code}
```python
df_enc_ord = pd.DataFrame( {'AGE': df['AGE'].map( mapping_dict_age) },
    index = df.index
)
```
:::


::: {.cell .markdown}
We can extend this approach to encode more than one ordinal feature. For example, let us consider the column `EDUC12R`, which includes the respondent’s answer to the question:

> Which best describes your education?
>
> 1.  High school or less
> 2.  Some college/assoc. degree
> 3.  College graduate
> 4.  Postgraduate study

:::

::: {.cell .code}
```python
df['EDUC12R'].value_counts()
```
:::

::: {.cell .markdown}
We can map both `AGE` and `EDUC12R` to ordinal-encoded columns in a new data frame:

:::


::: {.cell .code}
```python
mapping_dict_age = {'18-29': 1, 
                 '30-44': 2,
                 '45-65': 3,
                 '65+': 4}
mapping_dict_educ12r =  {'High school or less': 1,
                   'Some college/assoc. degree': 2,
                   'College graduate': 3,
                   'Postgraduate study': 4} 
df_enc_ord = pd.DataFrame( {
    'AGE': df['AGE'].map( mapping_dict_age) ,
    'EDUC12R': df['EDUC12R'].map( mapping_dict_educ12r) 
    },
    index = df.index
)
```
:::


::: {.cell .markdown}

Note that the order matters - the “High school or less” answer should have the smallest value, followed by “Some college/assoc. degree”, then “College graduate”, then “Postgraduate study”.

:::


::: {.cell .markdown}
Also note that missing values are still treated as missing (not mapped to some value) - this is going to be important, since we are going to design a distance metric that treats missing values sensibly:
:::


::: {.cell .code}
```python
df_enc_ord.isna().sum()
```
:::


::: {.cell .markdown}

There's one more important step before we can use our ordinal-encoded values with KNN.

Note that the values in the encoded columns range from 1 to the number of categories. For K nearest neighbors, the "importance" of each feature in determining the class label would be proportional to its scale (because the value of the feature is used directly in the distance metric). If we leave it as is, any feature with a larger range of possible values will be considered more "important!", i.e. would count more in the distance metric.

So, we will re-scale our encoded features to the unit interval. We can do this with the `MinMaxScaler` in `sklearn`. 

(Note: in general, you'd "fit" scalers etc. on only the training data, not the test data! In this case, however, the min and max in the training data is just due to our encoding, and will definitely be the same as the test data, so it doesn't really matter.)

:::


::: {.cell .code}
```python
scaler = MinMaxScaler()
 
# first scale in numpy format, then convert back to pandas df
df_scaled = scaler.fit_transform(df_enc_ord)
df_enc_ord = pd.DataFrame(df_scaled, columns=df_enc_ord.columns)
```
:::

::: {.cell .code}
```python
df_enc_ord.describe()
```
:::

::: {.cell .code}
```python
df_enc_ord['EDUC12R'].value_counts()
```
:::

::: {.cell .code}
```python
df_enc_ord.isna().sum()
```
:::

::: {.cell .markdown}
Later, you'll design a model with more ordinal features. For this initial demo, though, we'll stick to just those two - age and education - and continue to the next step.
:::


::: {.cell .markdown}
### Encode categorical features

:::


::: {.cell .markdown}
In the previous section, we encoded features that have a logical ordering.

Other categorical features, such as `RACE`, have no logical ordering. It would be wrong to assign an ordered mapping to these features. These should be encoded using **one-hot encoding**, which will create a new column for each unique value, and then put a 1 or 0 in each column to indicate the respondent's answer.

(Note: for features that have two possible values - binary features - either categorical encoding or one-hot encoding would be valid in this case!)


:::


::: {.cell .code}
```python
df['RACE'].value_counts()
```
:::


::: {.cell .markdown}

We can one-hot encode this column using the `get_dummies` function in `pandas`.

:::

::: {.cell .code}
```python
df_enc_oh = pd.get_dummies(df['RACE'], prefix='RACE', dtype=np.int32)
```
:::

::: {.cell .code}
```python
df_enc_oh.describe()
```
:::

::: {.cell .markdown}

Note that we added a `RACE` prefix to each column name - this prevents overlap between columns, e.g. if we also encoded another feature where "Other" was a possible answer. And, it helps us relate the new columns back to the original survey question that they answer.

:::


::: {.cell .markdown}
For this survey data, we want to preserve information about missing values - if a sample did not have a value for the `RACE` feature, we want it to have a NaN in all `RACE` columns. We can assign NaN to those rows as follows:
:::


::: {.cell .code}
```python
df_enc_oh.loc[df['RACE'].isnull(), df_enc_oh.columns.str.startswith("RACE_")] = float("NaN")
```
:::


::: {.cell .markdown}

Now, for respondents where this feature is not available, we have a NaN in all `RACE` columns:

:::


::: {.cell .code}
```python
df_enc_oh.isnull().sum()
```
:::



::: {.cell .markdown}
### Stack columns
:::


::: {.cell .markdown}
Now, we'll prepare our feature data, by column-wise concatenating the ordinal-encoded feature columns and the one-hot-encoded feature columns:

:::


::: {.cell .code}
```python
X = pd.concat([df_enc_oh, df_enc_ord], axis=1)
```
:::



::: {.cell .markdown}
### Get training and test indices
:::


::: {.cell .markdown}
We'll be working with many different subsets of this dataset, including different columns. 

So instead of splitting up the data into training and test sets, we'll get an array of training indices and an array of test indices using `ShuffleSplit`. Then, we can use these arrays throughout this notebook.
:::


::: {.cell .code}
```python
idx_tr, idx_ts = next(ShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 3).split(df['PRES']))
```
:::



::: {.cell .markdown}
I specified the state of the random number generator for repeatability, so that every time we run this notebook we'll have the same split. This makes it easier to discuss specific examples.
:::


::: {.cell .markdown}
Now, we can use the `pandas` function `.iloc` to get the training and test parts of the data set for any column.

For example, if we want the training subset of `y`:
:::

::: {.cell .code}
```python
y.iloc[idx_tr]
```
:::


::: {.cell .markdown}
or the test subset of `y`:
:::


::: {.cell .code}
```python
y.iloc[idx_ts]
```
:::


::: {.cell .markdown}
Here are the summary statistics for the training data:
:::

::: {.cell .code}
```python
X.iloc[idx_tr].describe()
```
:::

::: {.cell .markdown}
## Train a k nearest neighbors classifier
:::


::: {.cell .markdown}
Now that we have a target variable, a few features, and training and test indices, let's see what happens if we try to train a K nearest neighbors classifier.

:::


::: {.cell .markdown}
### Baseline: "prediction by mode"
:::


::: {.cell .markdown}

As a baseline against which to judge the performance of our classifier, let's find out the accuracy of a classifier that gives the majority class label (0) to all samples in our test set:

:::


::: {.cell .code}
```python
y_pred_baseline = np.repeat(0, len(y.iloc[idx_ts]))
accuracy_score(y.iloc[idx_ts], y_pred_baseline)
```
:::

::: {.cell .markdown}
A classifier trained on the data should do *at least* as well as the one that predicts the majority class label. Hopefully, we'll be able to do much better!

:::


::: {.cell .markdown}
### `KNeighborsClassifier` does not support data with NaNs
:::


::: {.cell .markdown}
We've previously seen the `sklearn` implementation of a `KNeighborsClassifier`. However, that won't work for this problem. If we try to train a `KNeighborsClassifier` on our data using the default settings, it will fail with the error message

    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

See for yourself:
:::



::: {.cell .code}
```python
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X.iloc[idx_tr], y.iloc[idx_tr])

```
:::

::: {.cell .markdown}
This is because we have many missing values in our data. And, as we explained previously, dropping rows with missing values is not a good option for this example.

:::

::: {.cell .markdown}

Although we cannot use the `sklearn` implementation of a `KNeighborsClassifier`, we can write our own. We need a few things:

-   a function that implements a distance metric
-   a function that accepts a distance matrix and returns the indices of the K smallest values for each row
-   a function that returns the majority vote of the training samples represented by those indices

and we have to be prepared to address complications at each stage!

:::


::: {.cell .markdown}
### Distance metric
:::

::: {.cell .markdown}

Let's start with the distance metric. Suppose we use an L1 distance computed over the features that are non-NaN for both samples:

:::

::: {.cell .code}
```python
def custom_distance(a, b):
  dif = np.abs(np.subtract(a,b))    # element-wise absolute difference
  # dif will have NaN for each element where either a or b is NaN
  l1 = np.nansum(dif, axis=1)  # sum of differences, treating NaN as 0
  return l1
```
:::


::: {.cell .markdown}
The function above expects a vector for the first argument and a matrix for the second argument, and returns a vector.

For example: suppose you pass a test point $x_t$ and a matrix of training samples where each row $x_0, \ldots, x_n$ is another training sample. It will return a vector $d_t$ with as many elements as there are training samples, and where the $i$th entry is the distance between the test point $x_t$ and training sample $x_i$.

To see how to this function is used, let\'s consider an example with a small number of test samples and training samples.

:::

::: {.cell .markdown}
Suppose we had this set of test data `a` (sampling some specific examples from the real data):

:::


::: {.cell .code}
```python
a_idx = np.array([10296, 510,4827,20937, 22501])
a = X.iloc[a_idx]
a
```
:::

::: {.cell .markdown}
and this set of training data `b`:

:::


::: {.cell .code}
```python
b_idx = np.array([10379, 4343, 7359,  1028,  2266, 131, 11833, 14106,  6682,  4402, 11899,  5877, 11758, 13163])
b = X.iloc[b_idx]
b
```
:::


::: {.cell .markdown}

We need to compute the distance from each sample in the test data `a`, to each sample in the training data `b`. 

We will set up a *distance matrix* in which to store the results. In the distance matrix, an entry in row $i$, column $j$ represents the distance between row $i$ of the test set and row $j$ of the training set.

So the distance matrix should have as many rows as there are test samples, and as many columns as there are training samples.

:::


::: {.cell .code}
```python
distances_custom = np.zeros(shape=(len(a_idx), len(b_idx)))
distances_custom.shape

```
:::


::: {.cell .markdown}

Now that we have the distance matrix set up, we're ready to fill it in with distance values. We will loop over each sample in the test set, and call the distance function passing that test sample and the entire training set.

Instead of a conventional `for` loop, we will use a [tqdm](https://github.com/tqdm/tqdm) `for` loop. This library conveniently "wraps" the conventional `for` loop with a progress part, so we can see our progress while the loop is running.

:::


::: {.cell .code}
```python
# the first argument to tqdm, range(len(a_idx)), is the list we are looping over
for idx in tqdm(range(len(a_idx)),  total=len(a_idx), desc="Distance matrix"):
  distances_custom[idx] = custom_distance(X.iloc[a_idx[idx]].values, X.iloc[b_idx].values)
```
:::


::: {.cell .markdown}
Let's look at those distances now:
:::


::: {.cell .code}
```python
np.set_printoptions(precision=2) # show at most 2 decimal places
print(distances_custom)
```
:::


::: {.cell .markdown}
### Find most common class of k nearest neighbors
:::


::: {.cell .markdown}
Now that we have this distance matrix, for each test sample, we can:

* get an array of indices from the *distance matrix*, sorted in order of increasing distance
* get the list of the K nearest neighbors as the first K elements from that list,
* from those entries - which are indices with respect to the distance matrix - get the corresponding indices in `X` and `y`,
* and then predict the class of the test sample as the most common value of `y` among the nearest neighbors.

:::


::: {.cell .code}
```python
k = 3
# array of indices sorted in order of increasing distance
distances_sorted = np.array([np.argsort(row) for row in distances_custom])
# first k elements in that list = indices of k nearest neighbors
nn_lists = distances_sorted[:, :k]
# map indices in distance matrix back to indices in `X` and `y`
nn_lists_idx = b_idx[nn_lists]
# for each test sample, get the mode of `y` values for the nearest neighbors
y_pred =  [y.iloc[nn].mode()[0] for nn in nn_lists_idx]
```
:::


::: {.cell .markdown}
### Example: one test sample
:::


::: {.cell .markdown}
For example, this was the first test sample:
:::


::: {.cell .code}
```python
X.iloc[[10296]]
```
:::


::: {.cell .markdown}
Here is its distance to each of the training samples in our "mini" training set:
:::


::: {.cell .code}
```python
distances_custom[0]
```
:::

::: {.cell .markdown}
and here's the sorted list of indices from that distance matrix - i.e. the index of the training sample with the smallest distance, the index of the training sample with the second-smallest distance, and so on.
:::


::: {.cell .code}
```python
distances_sorted[0]
```
:::

::: {.cell .markdown}
The indices (in the "mini" training sample) of the 3 nearest neighbors to this test sample are:
:::

::: {.cell .code}
```python
nn_lists[0]
```
:::



::: {.cell .markdown}
which corresponds to the following sample indices in the complete data `X`:
:::


::: {.cell .code}
```python
nn_lists_idx[0]
```
:::


::: {.cell .markdown}
So, its closest neighbors in the "mini" training set are:
:::


::: {.cell .code}
```python
X.iloc[nn_lists_idx[0]]
```
:::


::: {.cell .markdown}
and their corresponding values in `y` are:
:::


::: {.cell .code}
```python
y.iloc[nn_lists_idx[0]]
```
:::


::: {.cell .markdown}
and so the predicted label for the first test sample would be:
:::


::: {.cell .code}
```python
y.iloc[nn_lists_idx[0]].mode().values
```
:::


::: {.cell .markdown}
### Example: entire test set
:::


::: {.cell .markdown}
Now that we understand how our custom distance function works, let's compute the distance between every *test* sample and every *training* sample. 

We'll store the results in `distances_custom`.
:::


::: {.cell .code}
```python
distances_custom = np.zeros(shape=(len(idx_ts), len(idx_tr)))
distances_custom.shape
```
:::


::: {.cell .markdown}
To compute the distance vector for each test sample, loop over the indices in the *test* set:
:::


::: {.cell .code}
```python
for idx in tqdm(range(len(idx_ts)),  total=len(idx_ts), desc="Distance matrix"):
  distances_custom[idx] = custom_distance(X.iloc[idx_ts[idx]].values, X.iloc[idx_tr].values)
```
:::


::: {.cell .markdown}
Then, we can compute the K nearest neighbors using those distances:
:::


::: {.cell .code}
```python
k = 3

# get nn indices in distance matrix
distances_sorted = np.array([np.argsort(row) for row in distances_custom]) 
nn_lists = distances_sorted[:, :k]

# get nn indices in training data matrix
nn_lists_idx = idx_tr[nn_lists]

# predict using mode of nns
y_pred =  [y.iloc[nn].mode()[0] for nn in nn_lists_idx]
```
:::


::: {.cell .code}
```python
accuracy_score(y.iloc[idx_ts], y_pred)
```
:::


::: {.cell .markdown}

That is... not great.

:::


::: {.cell .markdown}
### Problems with our simple classifier
:::


::: {.cell .markdown}
The one-sample example we saw above is enough to illustrate some basic problems with our classifier, and to explain some of the reasons for its poor performance:

* the distance metric does not really tell us how *similar* two samples are, when there are samples with missing values, 
* and the way that ties are handled - when there are multiple samples in the training set with the same distance - is not ideal.

We'll discuss both of these, but we'll only fix the second one in this section. Part of *your* assignment will be to address the issue with the custom distance metric in your solution.
:::


::: {.cell .markdown}

In the example with the "mini" training and test sets, you may have noticed a problem: training sample 10379, which has all NaN values, has zero distance to *every* test sample according to our distance function. (Note that the first column in the distance matrix, corresponding to the first training sample, is all zeros.)

This means that this sample will be a "nearest neighbor" of *every* test sample! But, it's not necessarily *really* similar to those other test samples. We just *don't have any information* by which to judge how similar it is to other samples. These values are *unknown*, not *similar*.

The case with an all-NaN training sample is a bit extreme, but it illustrates how our simple distance metric is problematic in other situations as well. In general, when there are no missing values, for a pair of samples each feature is either *similar* or *different*. Thus a metric like L1 distance, which explicitly measures the extent to which features are *different*, also implicitly captures the extent to which features are *similar*. When samples can have missing values, though, for a pair of samples each feature is either *similar*, *different*, or *unknown* (one or both samples is missing that value). In this case, a distance metric that only measures the extent of *difference* (like L1 or L2 distance) does not capture whether the features that are not different are *similar* or *unknown*. (Our custom distance metric, which is an L1 distance, treats values that are *unknown* as if they are *similar* - neither one increases the distance.) Similarly, a distance metric that only measures the extent of *similarity* would not capture whether the features that are not similar are *different* or *unknown*.

So when there are NaNs, our custom distance metric does not quite behave the way we want - we want distance between two samples to decrease with more similarity, and to increase with more differences.  Our distance metric only considers difference, not similarity.

For example, consider these two samples from the original data:

:::


::: {.cell .code}
```python
pd.set_option('display.max_columns', 150)
disp_features = ['AGE8', 'RACE', 'REGION', 'SEX', 'SIZEPLAC', 'STANUM', 'EDUC12R', 'EDUCCOLL','INCOME16GEN', 'ISSUE16', 'QLT16', 'VERSION']
df.iloc[[0,1889]][disp_features]
```
:::


::: {.cell .markdown}
These two samples have some things in common:

* female
* from suburban California

but we don't know much else about what they have in common or what they disagree on. 

Our distance metric will consider them very similar, because they are identical with respect to every feature that is available in both samples.


:::


::: {.cell .code}
```python
custom_distance(X.iloc[[0]].values, X.iloc[[1889]].values)
```
:::



::: {.cell .markdown}
On the other hand, consider these two samples:
:::

::: {.cell .code}
```python
df.iloc[[0,14826]][disp_features]
```
:::


::: {.cell .markdown}
These two samples have many more things in common: 

* female
* Latino 
* age 18-24 
* no college degree
* income less then $30,000
* consider foreign policy to be the major issue facing the country
* consider "Has good judgment" to be the most important quality in deciding their presidential vote. 

However, they also have some differences:

* some college/associate degree vs. high school education or less
* suburban California vs. rural Oklahoma

so the distance metric will consider them *less* similar than the previous pair, even though they have a lot in common.
:::


::: {.cell .code}
```python
custom_distance(X.iloc[[0]].values, X.iloc[[14826]].values)
```
:::


::: {.cell .markdown}
A better distance metric will consider the level of disagreement between samples *and* the level of agreement. That will be part of your assignment - to write a new `custom_distance`.
:::


::: {.cell .markdown}
Now, let's consider the second issue - how ties are handled.

Notice that in the example with the "mini" training and test sets, for the first test sample, there was one sample with 0 distance and 3 samples with 0.33 distance. The three nearest neighbors are the sample with 0 distance, and the *first 2* of the 3 samples with 0.33 distance.

In other words: ties are broken in favor of the samples that happen to have lower indices in the data.

:::


::: {.cell .markdown}
On a larger scale, that means that some samples will have too much influence - they will appear over and over again as nearest neighbors, just because they are earlier in the data - while some samples will not appear as nearest neighbors at all simply because of this tiebreaker behavior.

If a sample is returned as a nearest neighbor very often because it happens to be closer to the test points than other points, that would be OK. But in this case, that's not what is going on.

For example, here are the nearest neighbors for the first 50 samples in the entire test set. Do you see any repetition?
:::


::: {.cell .code}
```python
print(nn_lists_idx[0:50])
```
:::

::: {.cell .markdown}
We find that these three samples appear very often as nearest neighbors:
:::


::: {.cell .code}
```python
X.iloc[[876, 10379,  1883]]
```
:::

::: {.cell .markdown}
But other samples that have the same distance - that are actually identical in `X`! - do not appear in the nearest neighbors list at all:
:::


::: {.cell .code}
```python
X[X['RACE_Hispanic/Latino'].eq(0) & X['RACE_Asian'].eq(0) & X['RACE_Other'].eq(0) 
  & X['RACE_Black'].eq(0) &  X['RACE_White'].eq(1)  
  & X['EDUC12R'].eq(1/3.0) & pd.isnull(X['AGE'])  ]
```
:::


::: {.cell .markdown}
A better tiebreaker behavior would be to randomly sample from neighbors with equal distance. Fortunately, this is an easy fix:

* We had been using `argsort` to get the K smallest distances to each test point. However, if there are more than K training samples that are at the minimum distance for a particular test point (i.e. a tie of more than K values, all having the minimum distance), `argsort` will return the first K of those in order of their index in the distance matrix (their order in `idx_tr`). 
* Now, we will use an alternative, `lexsort`, that sorts first by the second argument, then by the first argument; and we will pass a random array as the first argument:
:::


::: {.cell .code}
```python
k = 3
# make a random matrix
r_matrix = np.random.random(size=(distances_custom.shape))
# sort using lexsort - first sort by distances_custom, then by random matrix in case of tie
nn_lists = np.array([np.lexsort((r, row))[:k] for r, row in zip(r_matrix,distances_custom)])
nn_lists_idx = idx_tr[nn_lists]
y_pred =  [y.iloc[nn].mode()[0] for nn in nn_lists_idx]
```
:::


::: {.cell .markdown}
Now, we don't see nearly as much repitition of individual training samples among the nearest neighbors:
:::


::: {.cell .code}
```python
print(nn_lists_idx[0:50])
```
:::



::: {.cell .markdown}
Let's get the accuracy of *this* classifier, with the better tiebreaker behavior:
:::


::: {.cell .code}
```python
accuracy_score(y.iloc[idx_ts], y_pred)
```
:::


::: {.cell .markdown}

This classifier is less "fragile" - less sensitive to the draw of training data. 

(Depending on the random draw of training and test data, it may or may not have better performance for a particular split - but on average, across all splits of training and test data, it should be better.)

:::


::: {.cell .markdown}
### Use K-fold CV to select the number of neighbors
:::


::: {.cell .markdown}
In the previous example, we set the number of neighbors to 3, rather than letting this value be dictated by the data.
:::


::: {.cell .markdown}
As a next step, to improve the classifier performance, we can use K-fold CV to select the number of neighbors. Note that depending how we do it, this can be *very* computationally expensive, or it can be not much more computationally expensive than just fixing the number of neighbors ourselves.

The most expensive part of the algorithm is computing the distance to the training samples. This is $O(nd)$ for each test sample, where $n$ is the number of training samples and $d$ is the number of features. If we can make sure this computation happens only once, instead of once per fold, this process will be fast.

:::


::: {.cell .markdown}
Here, we pre-compute our distance matrix for *every* training sample:
:::


::: {.cell .code}
```python
# pre-compute a distance matrix of training vs. training data
distances_kfold = np.zeros(shape=(len(idx_tr), len(idx_tr)))

for idx in tqdm(range(len(idx_tr)),  total=len(idx_tr), desc="Distance matrix"):
  distances_kfold[idx] = custom_distance(X.iloc[idx_tr[idx]].values, X.iloc[idx_tr].values)
```
:::


::: {.cell .markdown}
Now, we'll use K-fold CV. 

In each fold, as always, we'll further divide the training data into validation and training sets. 

Then, we'll select the *rows* of the pre-computed distance matrix corresponding to the *validation* data in this fold, and the *columns* of the pre-computed distance matrix corresponding to the *training* data in this fold.
:::


::: {.cell .code}
```python
n_fold = 5
k_list = np.arange(1, 301, 10)
n_k = len(k_list)
acc_list = np.zeros((n_k, n_fold))

kf = KFold(n_splits=5, shuffle=True)

for isplit, idx_k in enumerate(kf.split(idx_tr)):

  print("Iteration %d" % isplit)

  # Outer loop: select training vs. validation data (out of training data!)
  idx_tr_k, idx_val_k = idx_k 

  # get target variable values for validation data
  y_val_kfold = y.iloc[idx_tr[idx_val_k]]

  # get distance matrix for validation set vs. training set
  distances_val_kfold   = distances_kfold[idx_val_k[:, None], idx_tr_k]

  # generate a random matrix for tie breaking
  r_matrix = np.random.random(size=(distances_val_kfold.shape))

  # loop over the rows of the distance matrix and the random matrix together with zip
  # for each pair of rows, return sorted indices from distances_val_kfold
  distances_sorted = np.array([np.lexsort((r, row)) for r, row in zip(r_matrix,distances_val_kfold)])

  # Inner loop: select value of K, number of neighbors
  for idx_k, k in enumerate(k_list):

    # now we select the indices of the K smallest, for different values of K
    # the indices in  distances_sorted are with respect to distances_val_kfold
    # from those - get indices in idx_tr_k, then in X
    nn_lists_idx = idx_tr[idx_tr_k[distances_sorted[:,:k]]]

    # get validation accuracy for this value of k
    y_pred =  [y.iloc[nn].mode()[0] for nn in nn_lists_idx]
    acc_list[idx_k, isplit] = accuracy_score(y_val_kfold, y_pred)
```
:::


::: {.cell .markdown}
Here's how the validation accuracy changes with number of neighbors:
:::


::: {.cell .code}
```python
plt.errorbar(x=k_list, y=acc_list.mean(axis=1), yerr=acc_list.std(axis=1)/np.sqrt(n_fold-1));

plt.xlabel("k (number of neighbors)");
plt.ylabel("K-fold accuracy");
```
:::



::: {.cell .markdown}
Using this, we can find a better choice for k (number of neighbors):
:::


::: {.cell .code}
```python
best_k = k_list[np.argmax(acc_list.mean(axis=1))]
print(best_k)
```
:::


::: {.cell .markdown}
Now, let's re-run our KNN algorithm using the entire training set and this `best_k` number of neighbors, and check its accuracy?
:::


::: {.cell .code}
```python
r_matrix = np.random.random(size=(distances_custom.shape))
nn_lists = np.array([np.lexsort((r, row))[:best_k] for r, row in zip(r_matrix,distances_custom)])
nn_lists_idx = idx_tr[nn_lists]
y_pred =  [y.iloc[nn].mode()[0] for nn in nn_lists_idx]
```
:::


::: {.cell .code}
```python
accuracy_score(y.iloc[idx_ts], y_pred)
```
:::

::: {.cell .markdown}
### Summarizing our basic classifier
:::

::: {.cell .markdown}

Our basic classifier:

* uses three features (age, race, and education) to predict a respondent's vote
* doesn't mind if there are NaNs in the data (unlike the `sklearn` implementation, which throws an error)
* uses a random tiebreaker if there are multiple training samples with the same distance to the test sample 
* uses the number of neighbors with the best validation accuracy, according to K-fold CV.

But, there are some outstanding issues:

* we have only used three features, out of many more available features.
* the distance metric only cares about the degree of disagreement (difference) between two samples, and doesn't balance it against the degree of agreement (similarity).

For this assignment, you will create an even better classifier by improving on those two issues.

:::



::: {.cell .markdown}
## Create a better classifier
:::

::: {.cell .markdown}

In the remaining sections of this notebook, you'll need to fill in code to:

* implement a custom distance metric
* encode more features
* implement feature weighting
* "train" and evaluate your final classifier, including K-Fold CV to select the best value for number of neighbors.

:::

::: {.cell .markdown}
### Create a better distance metric
:::


::: {.cell .markdown}

Your first task is to improve on the basic distance metric we used above. There is no one correct answer - there are many ways to compute a distance - but for full credit, your distance metric should satisfy the following criteria:

1. if two samples are identical, the distance between them should be zero.
2. as the extent of *difference* between two samples increases, the distance should increase.
3. as the extent of *similarity* between two samples increases, the distance should decrease.
4. if in a pair of samples one or both have a NaN value for a given feature, the similarity or difference of this feature is *unknown*. Your distance metric should compute a smaller distance for a pair of samples with many similarities (even if there is some small difference) than for a pair of samples with mostly unknown similarity.

You should also avoid explicit `for` loops inside the `custom_distance` function - use efficient `numpy` functions instead. Note that `numpy` includes many functions that are helpful when working with arrays that have NaN values, including mathematical functions like [sum](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html), [product](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html), [max](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html) and [min](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html), and logic functions like [isnan](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html).

:::

::: {.cell .markdown}

#### 📝 Specific requirements


**Function signature**: 

* Your `custom_distance` should accept a 1D array `a` (representing a single sample) as its first argument, and a 2D array `b` (representing a set of training samples) as its second argument. Then, it returns an array of distances from `a` (which is one row), to each row in `b` (which has multiple rows). 
* The array that is returned will have many *columns* as there are *rows* in `b`.
* Your `custom_distance` should also accept an optional `debug` argument, which defaults to `False`. If set to `True`, it will return some additional intermediate variables, as explained below.

**Missing values**: Your `custom_distance` function should *not* impute 0s or any other value in place of `NaN` values in either `a` or `b`. 

**Intermediate variables**: Your function should compute the following values, again using `a` against each row in `b`:

* `total_dif` = total magnitude of "disagreements"/"known dissimilarity" between `a` and each row in `b` (This is the L1 distance for known values)
* `total_nan` = total number of NaN/"unknown" values where either `a` *OR* the corresponding row of `b` (or both!) has a NaN

and you should use these (*not* only `total_dif`) in computing your distances. Also, when `debug = True` is passed to the function, you should return these intermediate variables in a dictionary, as shown in the example below.


:::


::: {.cell .markdown}
#### Implement your distance metric
:::

::: {.cell .code}
```python
# TODO - implement distance metric

def custom_distance(a, b, debug=False):

  # fill in your solution here!
  # you are encouraged to use efficient numpy functions where possible
  # refer to numpy documentation

  # you must compute these intermediate variables, and use them 
  # in computing your distances. Each of these should be a 1D `numpy` array
  # with as many elements as there are rows in `b`.
  total_dif = ...
  total_nan = ...

  # this is just a placeholder - your function shouldn't actually return 
  # all zeros ;)
  distances = np.zeros(b.shape[0])

  if debug:
    # if you are asked to return the intermediate variables too
    return distances, {'total_dif': total_dif, 'total_nan': total_nan}

  else:
    # the default case - don't return intermediate variables
    return distances
```
:::

::: {.cell .markdown}
#### Test cases for your distance metric
:::


::: {.cell .markdown}
You can use these test samples to check your work. (But, your metric should also satisfy the criteria in general - not only for these specific cases!)
:::

::: {.cell .markdown}
First criteria: if two samples are identical, the distance between them should be zero.
:::

::: {.cell .code}
```python
a = np.array([[0, 1, 0,      1, 0, 0.3]] )  # A0 - test sample
b = np.array([[0, 1, 0,      1, 0, 0.3]] )  # B0 - same as A0, should have 0 distance

```
:::

::: {.cell .code}
```python
distances_ex = np.zeros(shape=(len(a), len(b)))
for idx, a_i in enumerate(a):
  distances_ex[idx] = custom_distance(a_i, b)

print(distances_ex)
```
:::

::: {.cell .markdown}
Second criteria: as the extent of *difference* between two samples increases, the distance should increase.
:::

::: {.cell .markdown}
These should have *increasing* distance:
:::

::: {.cell .code}
```python
a = np.array([[0, 1, 0,      1, 0, 0.3]] )  # A0 - test sample
b = np.array([[0, 1, 0,      1, 0, 0.3],              # B0 - same as A0, should have 0 distance
              [0, 1, 0,      1, 0, 0.5],              # B1 - has one small difference, should have larger distance than B0
              [0, 1, 0,      1, 0, 1  ],              # B2 - has more difference, should have larger distance than B1
              [0, 0, 0,      1, 0, 0  ],              # B3 - has even more difference
              [1, 0, 1,      0, 1, 0  ]])             # B4 - has the most difference
```
:::

::: {.cell .code}
```python
distances_ex = np.zeros(shape=(len(a), len(b)))
for idx, a_i in enumerate(a):
  distances_ex[idx] = custom_distance(a_i, b)

print(distances_ex)
```
:::

::: {.cell .markdown}
These should have *decreasing* distance:
:::

::: {.cell .code}
```python
a = np.array([[0, 1, 0, 1, 0, 1]] )            # A0 - test sample
b = np.array([[1, 0, 1, 0, 1, 0],              # B0 - completely different, should have large distance
              [1, 0, 1, 0, 1, np.nan],         # B1 - less difference than B0, should have less distance
              [1, 0, 1, 0, np.nan, np.nan]])   # B2 - even less difference than B1, should have less distance
```
:::

::: {.cell .code}
```python
distances_ex = np.zeros(shape=(len(a), len(b)))
for idx, a_i in enumerate(a):
  distances_ex[idx] = custom_distance(a_i, b)

print(distances_ex)
```
:::

::: {.cell .markdown}
Third criteria: as the extent of *similarity* between two samples increases, the distance should decrease.
:::

::: {.cell .markdown}
These should have *increasing* distance:
:::

::: {.cell .code}
```python
a = np.array([[0, 1, 0, 1, 0, 0.3]] )  # A0 - test sample
b = np.array([[0, 1, 0, 1, 0, 0.3],              # B0 - same as A0, should have 0 distance
              [0, 1, 0, 1, 0, np.nan],           # B1 - has less similarity than B0, should have larger distance
              [0, 1, 0, 1, np.nan, np.nan],      # B2 - has even less similarity, should have larger distance
              [0, np.nan, np.nan, np.nan, np.nan, np.nan]])     # B3 - has least similarity, should have larger distance
```
:::

::: {.cell .code}
```python
distances_ex = np.zeros(shape=(len(a), len(b)))
for idx, a_i in enumerate(a):
  distances_ex[idx] = custom_distance(a_i, b)

print(distances_ex)
```
:::

::: {.cell .markdown}
Fourth criteria: if in a pair of samples one or both have a NaN value for a given feature, the similarity or difference of this feature is *unknown*. Your distance metric should compute a smaller distance for a pair of samples with many similarities (even if there is some small difference) than for a pair of samples with mostly unknown similarity.
:::

::: {.cell .markdown}
These should have *increasing* distance:
:::

::: {.cell .code}
```python
a = np.array([[0, np.nan, 0, 1, np.nan, 0.3]] )  # A0 - test sample
b = np.array([[0, np.nan, 0, 1, 0,      0.5],                # B0 - three similar features, one small difference
              [0, np.nan, np.nan, np.nan, np.nan, np.nan]])  # B1 - much less similarity than B0, should have larger distance

```
:::

::: {.cell .code}
```python
distances_ex = np.zeros(shape=(len(a), len(b)))
for idx, a_i in enumerate(a):
  distances_ex[idx] = custom_distance(a_i, b)

print(distances_ex)
```
:::


::: {.cell .markdown}
### Encode more features
:::


::: {.cell .markdown}

Our basic classifier used three features: age, race, and education. But there are many more features in this data that may be predictive of vote:

-   More demographic information: `INCOME16GEN`, `MARRIED`, `RELIGN10`, `ATTEND16`, `LGBT`, `VETVOTER`, `SEX`
-   Opinions about political issues and about what factors are most important in determining which candidate to vote for: `TRACK`, `SUPREME16`, `FINSIT`, `IMMWALL`, `ISIS16`, `LIFE`, `TRADE16`, `HEALTHCARE16`, `GOVTDO10`, `GOVTANGR16`, `QLT16`, `ISSUE16`, `NEC`

in addition to the features `AGE`, `RACE`, and `EDUC12R`.

You will try to improve the model by adding some of these features.


(Note that we will *not* use questions that directly ask the participants how they feel about individual candidates, or about their party affiliation or political leaning. These features are a close proxy for the target variable, and we're going to assume that these are not available to the model.)



:::


::: {.cell .markdown}

Refer to the PDF documentation to see the question and the possible answers corresponding to each of these features. You may also choose to do some exploratory data analysis, to help you understand these features better.

For your convenience, here are all the possible answers to those survey questions:

:::


::: {.cell .code}
```python
features = ['INCOME16GEN', 'MARRIED', 'RELIGN10', 'ATTEND16', 'LGBT', 'VETVOTER', 
            'SEX', 'TRACK', 'SUPREME16',  'FINSIT', 'IMMWALL', 'ISIS16', 'LIFE', 
            'TRADE16', 'HEALTHCARE16', 'GOVTDO10', 'GOVTANGR16', 'QLT16', 
            'ISSUE16', 'NEC']

for f in features:
  print(f)
  print(df[f].value_counts())
  print("***************************************************")

```
:::


::: {.cell .markdown}


#### 📝 Specific requirements

It is up to you to decide which features to include in your model, from the features in the following list: `INCOME16GEN`, `MARRIED`, `RELIGN10`, `ATTEND16`, `LGBT`, `VETVOTER`, `SEX`, `TRACK`, `SUPREME16`, `FINSIT`, `IMMWALL`, `ISIS16`, `LIFE`, `TRADE16`, `HEALTHCARE16`, `GOVTDO10`, `GOVTANGR16`, `QLT16`, `ISSUE16`, `NEC`, `AGE`, `RACE`, `EDUC12R`.


However, you must encode at least eight features, including:

*  at least four features that are encoded using an ordinal encoder because they have a logical order (and you should include an explicit mapping for these), and
* at least four features that are encoded using one-hot encoding because they have no logical order.

Binary features - features that can take on only two values - "count" toward either category.

(If you decide to use the features I used above, they do "count" as part of the four. For example, you could use age, education, and two additional ordinal-encoded features, and race and three other one-hot-encoded features.)

You will also be required to justify your decision, specifically with respect to the "missing values" aspect. Make sure you are including some informative features from *each* of the survey versions. After feature encoding, you should compute the number of non-missing values per row, and report some summary statistics about this value.

:::


::: {.cell .markdown}
#### Encode ordinal features
:::


::: {.cell .markdown}

In the following cells, prepare your ordinal encoded features as demonstrated in the "Prepare data > Encode ordinal features" section earlier in this notebook.

Use at least four features that are encoded using an ordinal encoder. (You can choose which features to include, but they should be either binary features, or features for which the values have a logical ordering that should be preserved in the distance computations!)

Also:

* Save the ordinal-encoded columnns in a data frame called `df_enc_ord`.
* You should explicitly specify the mappings for these, so that you can be sure that they are encoded using the correct logical order.
* For some questions, there is also an "Omit" answer - if a respondent left that question blank on the questionnaire, the value for that question will be "Omit". Since "Omit" has no logical place in the order, we're going to treat these as missing values: don't include "Omit" in your `mapping_ord` dictionary, and then these Omit values will be encoded as NaN.
* Make sure to scale each column to the range 0-1, as demonstrated in the "Prepare data > Encode ordinal features" section earlier in this notebook.

:::


::: {.cell .code}
```python
# TODO - encode ordinal features

# set up mapping dictionary and list of features to encode with ordinal encoding

# use map to get the encoded columns, save in df_enc_ord
df_enc_ord = ...

# scale each column to the range 0-1
df_enc_ord = 

```
:::


::: {.cell .markdown}

Look at the encoded data to check your work:

:::


::: {.cell .code}
```python
df_enc_ord.describe()
```
:::



::: {.cell .markdown}
#### Encode categorical features
:::



::: {.cell .markdown}

In the following cells, prepare your categorical encoded features as demonstrated in the "Prepare data > Encode categorical features" section earlier in this notebook.

Use at least four features that are encoded using an categorical encoder. (You can choose which features to include, but they should be either binary features, or features for which the values do *not* have a logical ordering that should be preserved in the distance computations!)

Also:

* Save the categorical-encoded columnns in a data frame called `df_enc_oh`.
* For some questions, there is also an "Omit" answer - if a respondent left that question blank on the questionnaire, the value for that question will be "Omit". We're going to treat these as missing values. Before encoding the NaN values, you should drop the column corresponding to the "Omit" value from the data frame.

:::


::: {.cell .code}
```python
# TODO - encode categorical features


# use get_dummies to get the encoded columns, stack and save in df_enc_oh
df_enc_oh = ...

# drop the Omit columns, if any of these are in the data frame
df_enc_oh.drop(['ISSUE16_Omit', 'QLT16_Omit', 'TRACK_Omit','IMMWALL_Omit','GOVTDO10_Omit'], 
                axis=1, inplace=True, errors='ignore')

# if a respondent did not answer a question, make sure they have NaN in all the columns corresonding to that question
```
:::

::: {.cell .markdown}
#### Stack columns
:::


::: {.cell .markdown}
Now, we'll create a combined data frame with all of the encoded features:
:::


::: {.cell .code}
```python
X = pd.concat([df_enc_oh, df_enc_ord], axis=1)
```
:::


::: {.cell .code}
```python
X.describe()
```
:::

::: {.cell .markdown}
#### TODO - describe your choice of features to encode

:::

::: {.cell .markdown}

In a text cell, discuss the features you chose to include. 

Also show, for each version of the survey (1, 2, 3, 4, 5), which of the features you included are on that survey?

:::


::: {.cell .markdown}
### Feature weighting
:::


::: {.cell .markdown}

Because the K nearest neighbor classifier weights each feature equally in the distance metric, including features that are not relevant for predicting the target variable can actually make performance worse.

To improve performance, we will use feature weights, so that more important features are scaled up and less important features are scaled down.

:::


::: {.cell .markdown}

#### 📝 Specific requirements


There are many options for feature selection or feature weighting - there isn't one right answer here! In our lesson on feature selection/weighting, we discussed two parts to the problem of identifying the best subset of features:

* **Search**: the strategy you use to determine the features or feature subsets to evaluate.
* **Evaluate**: the approach you use to evaluate the "goodness" of a feature or feature subset. Since this dataset has the added complication of missing values, you must also consider how you handle missing values in your evaluation. 

For this assignment, you will use a naive search strategy (score each feature independently). But, 

* First, divide the training set into a training and validation subset (single hold-out validation set). 
* I want you to consider **two** different scoring functions of your choice - i.e., compute two sets of "goodness" scores. 
* When computing scores, you should not impute 0s or any other value for `NaN` values in the data. Instead, you should compute the score for a feature using only the rows in the training data where *that* feature is not missing.

Then, use the hold-out validation set to decide which of the two scoring functions to use. For each scoring function,

* "Fit" a KNN model (with your custom distance function, random tie break, etc. as discussed above) on the training subset of the weighted feature data.
* Evaluate the model by computing its accuracy score on the held-out validation subset.

Also, for full credit,

* Your solution should not assign the same weight to all features.
* Your solution should not weight highly the features that are least useful for predicting the target variable.
* Your solution *should* weight highly the features are are most useful for predicting the target variable.
* Your implementation should satisfy the requirements above generally, not only for this specific data. (It will be evaluated on other data.)
* Your solution must be well justified.

:::


::: {.cell .markdown}

In the following cell and additional code cells as needed, implement feature weighting following the requirements described above, and return the results in `X_trans`:

* `X_trans` should have the same dimensions of `X`, but instead of each column being in the range 0-1, each column will be scaled according to its importance (more important features will be scaled up, less important features will be scaled down). You should create a variable `feat_wt` which has a weight for every feature in `X`. Then, you'll multiply `X` by `feat_wt` to get `X_trans`.

Some important notes:

* The goal is to write code for feature weighting, not to find it by manual inspection! Don't hard-code any values.
* Although `X_trans` will include all rows of the data, you should not use the test data in the process of finding `feat_wt`! Feature selection and feature weighting are considered part of model fitting, and so only the training data may be used in this process.
* You are free to use an `sklearn` function to compute feature scores, but make sure you understand what it does and are sure it is a good fit for the data and the model.

:::



::: {.cell .code}
```python
# TODO - feature weighting

# Xtr_fw, Xvl_fw, ytr_fw, yvl_fw = train_test_split(...) # divide training data intro training and validation split

# score_1 = ... # array of scores per column for first scoring function
# score_2 - ... # array of scores per column for second scoring function

# fit KNN model using score_1 to weight features
# acc_1 = ...   # accuracy of KNN model on validation set (weighted using score_1)

# fit KNN model using score_2 to weight features
# acc_2 = ...   # accuracy of KNN model on validation set (weighted using score_2)

# feat_wt = ... # final feature weights (whichever of score_1 or score_2 is better)
# X_trans = X.multiply(feat_wt)
```
:::


::: {.cell .markdown}
Check your work:
:::


::: {.cell .code}
```python
X_trans.describe()
```
:::



::: {.cell .markdown}
#### TODO - describe your approach to feature weighting
:::


::: {.cell .markdown}
In a text cell, describe the approach you used for feature weighting. Your answer should include the following parts, in paragraph form:

* Describe the two alternative scoring functions you used to evaluate the "goodness" of a feature or feature subset. Why did you think these might be well suited for *this data* and *this model*?
* Discuss the results on the held-out validation set - which scoring function had better performance? (You must refer to specific values from your evaluation.)

:::

::: {.cell .markdown}
### Evaluate final classifier
:::


::: {.cell .markdown}

Finally, you'll repeat the process of finding the best number of neighbors using K-fold CV, with 
your "transformed" data (`X_trans`) and your new custom distance metric.

Then, you'll evaluate the performance of your model on the *test* data, using that optimal number of neighbors.

:::



::: {.cell .code}
```python
# TODO - evaluate - pre-compute distance matrix of training vs. training data

distances_kfold = ...

```
:::



::: {.cell .code}
```python
# TODO - evaluate - use K-fold CV, fill in acc_list 

n_fold = 5
k_list = np.arange(1, 301, 10)
n_k = len(k_list)
acc_list = np.zeros((n_k, n_fold))

# use this random state so your results will match the auto-graders'
kf = KFold(n_splits=5, shuffle=True, random_state=3)

for isplit, idx_k in enumerate(kf.split(idx_tr)):

  # Outer loop

  for idx_k, k in enumerate(k_list):

    # Inner loop

    acc_list[idx_k, isplit] = ...
```
:::


::: {.cell .markdown}
See how the validation accuracy changes with number of neighbors:
:::


::: {.cell .code}
```python
plt.errorbar(x=k_list, y=acc_list.mean(axis=1), yerr=acc_list.std(axis=1)/np.sqrt(n_fold-1));

plt.xlabel("k (number of neighbors)");
plt.ylabel("K-fold accuracy");
```
:::



::: {.cell .markdown}
Find the best choice for k (number of neighbors) using the "highest validation accuracy" rule:
:::


::: {.cell .code}
```python
# TODO - evaluate - find best k
best_k = ...
```
:::


::: {.cell .markdown}
Finally, re-run our KNN algorithm using the entire training set and this `best_k` number of neighbors. Check its accuracy on the test data.
:::


::: {.cell .code}
```python
# TODO - evaluate - find accuracy
# compute distance matrix for test vs. training data
# use KNN with best_k to find y_pred for test data
y_pred = ...
# compute accuracy
acc = ...
```
:::


::: {.cell .code}
```python
print(acc)
```
:::
