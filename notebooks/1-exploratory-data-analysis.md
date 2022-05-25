---
title:  'Exploratory Data Analysis'
author: 'Fraida Fund'
---


::: {.cell .markdown}
# Exploratory data analysis

_Fraida Fund_

:::

::: {.cell .markdown}

## In this notebook


In this notebook:

* We practice using `pandas` to read in and manipulate a data set. (We won't have a separate tutorial on `pandas` - we will learn basic `pandas` techniques as we need them.)
* We learn a basic "recipe" for exploratory data analysis and apply it to an example


:::

::: {.cell .markdown}
## Introduction

:::

::: {.cell .markdown}

The first step in applying machine learning to a real problem is *finding* or *creating* an appropriate data set with which to train your model.

:::

::: {.cell .markdown}

### What makes data "good"?


What makes a good data set?

*  **Size**: the more *samples* are in the data set, the more examples your machine learning model will be able to learn from, and the better it will do. Often, a simple machine learning model trained on a large data set will outperform a "fancy" model on a small data set.
*  **Quality**: Are there *predictive* features in the data? Are no values (or very few values) missing, noisy, or incorrect? Is the scenario in which the data collected similar to the scenario in which your model will be used? These are examples of questions that we might ask to evaluate the quality of a data set.

:::

::: {.cell .markdown}

One of the most important principles in machine learning is: **garbage in, garbage out**. If the data you use to train a machine learning model is problematic, or not well suited for the purpose, then even the best model will produce useless predictions.

:::
::: {.cell .markdown}

### Purpose of exploratory data analysis


Once we have identified one or more candidate data sets for a particular problem, we perform some *exploratory data analysis*. This process helps us

* detect and possibly correct mistakes in the data
* check our assumptions about the data
* identify potential relationships between features
* assess the direction and rough size of relationships between features and the target variable

Exploratory data analysis is important for understanding whether this data set is appropriate for the machine learning task at hand, and if any extra cleaning or processing steps are required before we use the data.


:::

::: {.cell .markdown}

## "Recipe" for exploratory data analysis


We will practice using a basic "recipe" for exploratory data analysis.

1.  Set down *expectations* about the data
2.  Load data and check that it is loaded correctly
3.  Inspect the data to make sure it is consistent with your expectations ("sanity checks"), and clean or filter the data if needed
4.  Explore relationships in the data to identify good candidate features and target variables

Every exploratory data analysis is different, as specific characteristics of the data may lead you to explore different things in depth. However, this "recipe" can be a helpful starting point.

:::

::: {.cell .markdown}

## Example: Brooklyn Bridge pedestrian data set 


:::


::: {.cell .markdown}


The Brooklyn Bridge is a bridge that connects Brooklyn and Manhattan. It supports vehicles, pedestrians, and bikers.

![](https://brooklyneagle.com/wp-content/uploads/2019/01/7-Brooklyn-Bridge-pedestrians-in-bike-lane-to-right-of-white-stripe-January-2019-photo-by-Lore-Croghan-600x397.jpg)

:::


::: {.cell .markdown}

Support you are developing a machine learning model to predict the volume of pedestrian traffic on the Brooklyn Bridge. There is a dataset available that you think may be useful as training data: [Brooklyn Bridge Automated Pedestrian Counts dataset](https://data.cityofnewyork.us/Transportation/Brooklyn-Bridge-Automated-Pedestrian-Counts-Demons/6fi9-q3ta), from the NYC Department of Transportation.

We will practice applying the "recipe" for exploratory data analysis to this data.

We will use the `pandas` library in Python, which includes many powerful utilities for managing data. You can refer to the [`pandas` reference](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) for more details on the `pandas` functions used in this notebook.


:::


::: {.cell .markdown}


### Set down *expectations* about the data

The first step is to codify your expectations about the data *before* you look at it:

* Read about *methodology* and *data codebook*
* How many rows and columns are in the data? 
* What does each variable mean? What units are data recorded in? What is the expected range or typical value for each column?
* What variables do you think could be used as target variable? What variables could be used as features from which to learn?
* How was data collected? Identify sampling issues, timeliness issues, fairness issues, etc.


:::


::: {.cell .markdown}




For the Brooklyn Bridge dataset, you can review the associated documentation on the NYC Data website:

* [NYC Data Website](https://data.cityofnewyork.us/Transportation/Brooklyn-Bridge-Automated-Pedestrian-Counts-Demons/6fi9-q3ta)
* [Data dictionary](https://data.cityofnewyork.us/api/views/6fi9-q3ta/files/845905ea-21d4-4ec7-958a-a1a09214513d?download=true&filename=Brooklyn_Bridge_Automated_Pedestrian_Counts_Demonstration_Project_data_dictionary.xlsx)


:::

::: {.cell .markdown}

### Load data and check that it is loaded correctly

The next step is to load the data in preparation for our exploratory data analysis. Then, we'll check that it is loaded correctly. 

Some examples of the things we'll look for include:

* Does the `DataFrame` have the correct number of rows and columns (consistent with our expectations from the first step)?
* Is the first row of "data" in the `DataFrame` real data, or is it column labels that were misinterpreted as data? (Similarly, are the column labels actually labels, or are they the first row of data?)
* Are the data types of every column consistent with our expectations?

At this stage, we might also do some very basic manipulation of the data - for example, compute some fields that are derived directly from other fields. (For example, suppose you have a "distance" field in miles and you wanted to convert it to meters - you could do that here!)

:::

::: {.cell .markdown}

First, we will import some useful libraries:

* In Python - libraries add powerful functionality
* You can import an entire library (`import foo`) or part (`from foo import bar`)
* You can define a nickname, which you will use to call functions of these libraries (many libraries have "conventional" nicknames)


`pandas` is a popular Python library for working with data. It is conventionally imported with the `pd` nickname.

:::

::: {.cell .code}
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set up notebook to show all outputs, not only last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
:::



::: {.cell .markdown}

Now we are ready to read in our data!

The main type of data structure in `pandas` is a `DataFrame`, which organizes data into a 2D table, like a spreadsheet. Unlike a `numpy` array, however, each column in a `DataFrame` can have different data types - for example, you can have a string column, an integer column, and a float column all in the same `DataFrame`. 

(The other major type of data in `pandas` is a `Series`, which is like a 1D array- any individual row or column from a `DataFrame` will be a `Series`.)

:::

::: {.cell .markdown}

You *can* create a `DataFrame` or a `Series` "by hand" - for example, try

```python
pd.Series([1,2,3,99])
```

or 


```python
pd.DataFrame({'fruit': ['apple', 'banana', 'kiwi'], 'cost': [0.55, 0.99, 1.24]})
```

:::


::: {.cell .markdown}

But usually, we'll read in data from a file.

Our data for this Brooklyn Bridge example is in CSV format, so will use the `read_csv` function in `pandas` to read in our data.  This function accepts a URL or a path to a file, and will return our data as a `DataFrame`.

Function documentation: [pandas reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

```python
pandas.read_csv(filepath_or_buffer, 
	sep=',', header='infer', 
	names=None,
	...)
```

`read_csv` is for "flat" text files, where each data point is on another row, and the fields in a row are separated by some delimiter (e.g. comma). Other pandas functions exist for loading other kinds of data (read from database, Excel file, etc.)


:::



::: {.cell .code}
```python
url = 'https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(url)
```
:::



::: {.cell .markdown}

We will want to verify that the data was loaded correctly. For *tabular* data, we can start by looking at the first few rows of data or the last few rows of data with the `head` and `tail` functions, respectively. (For data that is not tabular, such as image, text, or audio data, we would similarly start by looking at some samples.)

:::

::: {.cell .code}

```python
df.head()
```
:::


::: {.cell .code}

```python
df.tail()
```
:::

::: {.cell .markdown}

We can also get a few random rows:

:::

::: {.cell .code}

```python
df.sample(5)
```
:::



::: {.cell .markdown}

Looking at some rows can help us spot obvious problems with data loading. For example, suppose we had tried to read in the data using a tab delimiter to separate fields on the same row, instead of a comma.

:::

::: {.cell .code}
```python
df_bad  = pd.read_csv(url, sep='\t')
df_bad.head()
```
:::

::: {.cell .markdown}

This "bad" version of the `DataFrame` has only a single column (because it believes tabs are used to separate fields in the same row, when actually commas are used). The variable names are combined together into one long column name. By looking at the first few rows of data, we can spot this obvious error.

Here is another example of a "bad" `DataFrame`. Suppose we tell `read_csv` that the data file itself does not have a 
header row at the top, with column names in it; instead, we supply column names ourselves. 

:::

::: {.cell .code}
```python
col_names = ["hour_beginning", "location", "Pedestrians", "Towards Manhattan", 
	"Towards Brooklyn", "weather_summary", "temperature", "precipitation", 
	"lat", "long", "events", "Location1"]
df_bad  = pd.read_csv(url, header=None, names=col_names)
df_bad.head()
```
:::


::: {.cell .markdown}

In this example, the first row in the file *is* actually a column header, and we mistakenly read it in as data. (A similar problem can occur in reverse - if we told `read_csv` that the first row *is* a header when it is not, then our "column labels" would actually be the first row of data.)

:::



::: {.cell .markdown}


We should always check the shape of the data frame - the number of rows and columns. This, too, should be checked against our assumptions about the data (in this case, what we know from the NYC Data website.)

:::

::: {.cell .code}

```python
df.shape
```
:::


::: {.cell .markdown}

Check the names of the columns and their data types:

:::

::: {.cell .code}

```python
df.columns
df.dtypes
```
:::

::: {.cell .markdown}

The main data types we'll see most often are `int64` (integer), `float64` (numeric), `bool` (True or False), or `object` (which includes string). 

:::

::: {.cell .markdown}

We can also get a quick summary with `info()`;

:::

::: {.cell .code}

```python
df.info()
```
:::



::: {.cell .markdown}

`pandas` infers the data type of each column automatically from the contents of the data.

If the data type of a column is not what you expect it to be, this can often be a signal that the data needs cleaning. For example, if you expect a column to be numeric and it is read in as non-numeric, this indicates that there are probably some samples that include a non-numeric value in that column. (The [NYC Data website](https://data.cityofnewyork.us/Transportation/Brooklyn-Bridge-Automated-Pedestrian-Counts-Demons/6fi9-q3ta) indicates what type of data *should* be in each column, so you should reference that when checking this output. )

:::

::: {.cell .markdown}

We have a date/time column (`hour_beginning`) that was read in as a string. Let's take a closer look at that. We can get one column of data either using a notation like a dictionary, as in

```python
df['hour_beginning']
```

or using class attribute-like notation, as in

```python
df.hour_beginning
```

(either one returns exactly the same thing!) (Note that if the column name includes spaces, you can only use the notation with the brackets, since it encloses the column name in quotes.)

:::

::: {.cell .markdown}


`pandas` includes a `to_datetime` function to convert this string to a "native" date/time format, so we can use that now:

:::


::: {.cell .code}

```python
df['hour_beginning'] = pd.to_datetime(df['hour_beginning'])
df.info()
```
:::


::: {.cell .markdown}

You may notice that the `hour_beginning` variable includes the full date and time in one field. For our analysis, it would be more useful to have separate fields for the date, month, day of the week, and hour.

We can create these additional fields by assigning the desired value to them directly - then, observe the effect:

:::

::: {.cell .code}
```python
df['hour'] = df['hour_beginning'].dt.hour
df['month'] = df['hour_beginning'].dt.month
df['date'] = df['hour_beginning'].dt.date
df['day_name'] = df['hour_beginning'].dt.day_name()

df.head()
```
:::

::: {.cell .markdown}


### Inspect (and possibly clean/filter) the data

Now we are ready to inspect the data.

Some examples of the things we'll look for include:

* Are there missing values? There may be rows *in* the data where some or all fields are missing (which can be denoted as None, NaN, or even 0 or -1 - which can be misleading when 0 or -1 are also valid values for that field.) There may also be rows *not in* the data, that we expect *should be* in the data.
* For numeric fields: Is the min and max of each field consistent with our expectation? Is the median consistent with our expectation?
* For non-numeric fields: Are the number of unique values in each field consistent with our expectations? Are the values of the factor levels (where these can reasonably be assessed) described consistently throughout the data?
* Are the relationships *between* variables consistent with our expectations? (We can evaluate this visually, and also by looking at summary statistics.)
* If the data is a time series, is the trend of each variable over time consistent with our expectations?

For many of these "sanity checks", we will need some *domain knowledge*. It's hard to have reasonable expectations about the values in the data if you do not understand the topic that the data relates to.

:::


::: {.cell .markdown}


#### Check whether data is complete

:::




::: {.cell .markdown}

Let us start by checking whether the data is complete. First, we'll check whether there are any rows in the data where some or all fields are missing.

We can see the number of missing values in each column by summing up all the instances where the `isnull` function returns a True value:

:::



::: {.cell .code}
```python
df.isnull().sum()
```
:::

::: {.cell .markdown}

(Note that this only tells us about missing values that are explicitly denoted as such - for example, explicit `NaN` values. If a missing value is coded as something else - like a 0 or -1 - we wouldn't know unless we noticed an unusually high frequency of 0 or -1 values.)

:::


::: {.cell .markdown}

We notice that the majority of rows are missing a value in the `events` field, which is used to mark dates that are holidays or other special events. This is reasonable, since most dates do not have any remarkable events. 

Let's look at the rows that *do* have a value in the `events` field. To filter a dataframe, we'll use the `.loc[]` operator. This accepts either an index (for example, we can do `df.loc[0]` to see the first record in the dataframe), an array of indices (for example, `df.loc[[0,1,2]]`), or an array of boolean values the length of the entire dataframe. That's what we'll use here.


:::

::: {.cell .code}
```python
df.loc[df['events'].notnull()]
```
:::

::: {.cell .markdown}



We also notice a small number of rows missing weather information. It's not clear why these are missing. Let's take a closer look at some of those rows, by *filtering* the dataframe to only rows that meet a specific condition - in this case, that the `temperature` field is missing.


:::


::: {.cell .code}

```python
df.loc[df.temperature.isnull()]
```
:::


::: {.cell .markdown}

We can see that for these particular instances, all of the weather information is missing. There's no obvious reason or pattern. We'll deal with these soon, when we try to clean/filter the data.

:::


::: {.cell .markdown}

Before we do that, though, let's check for the *other* kind of missing data: rows that are missing completely, that we expect *should* be present. 

In this example, the data is a time series, and we expect that there is exactly one row of data for every single hour over the time period in which this data was collected.

:::



::: {.cell .markdown}

Let's see if the data is complete, or if there are gaps in time.  

First, we will use [`pd.date_range`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html) to get the list of hour intervals that we expect to find in the dataset. Then, we will find the difference between this list and the actual list of hour intervals in the dataset - these are missing intervals.


:::


::: {.cell .code}
```python
# get beginning and end of date range
min_dt = df.hour_beginning.min()
max_dt = df.hour_beginning.max()
print(min_dt)
print(max_dt)
```
:::

::: {.cell .code}
```python
expected_range = pd.date_range(start = min_dt, end = max_dt, freq='H' )
expected_range
```
:::



::: {.cell .code}
```python
# then identify the missing hours
missing_hours = expected_range.difference(df['hour_beginning'])
print(missing_hours)
```
:::


::: {.cell .markdown}

We had the expected number of rows (the output of `shape` matched the description of the data on the NYC Data website), but the data seems to be missing samples from August 2018 through December 2018, which is worth keeping in mind if we decide to use it:

:::


::: {.cell .code}
```python
pd.unique(missing_hours.date)
```
:::


::: {.cell .markdown}

Let's also check if any hour appears more than once in the data. We can use `pandas`'s `value_counts` function to find out how many times each unique value appears in the data:
:::


::: {.cell .code}
```python
df['hour_beginning'].value_counts()
```
:::

::: {.cell .markdown}

It looks like at least one hour appears twice in the data, which is unexpected! Let's use filtering again to find out all of the instances where that occurs:

:::

::: {.cell .code}
```python
hour_counts = df['hour_beginning'].value_counts()
hour_counts.loc[hour_counts > 1]
```
:::

::: {.cell .markdown}

It seems to happen exactly once. Let's filter the dataframe to find the rows corresponding to the duplicate day.

Here's a useful clue - we can see that this hour appears twice because the clock is shifted for Daylight Savings time. (It's not clear why there is no duplicate hour for that same event in 2017. Perhaps only one of those hours is recorded.) 


:::

::: {.cell .code}
```python
df.loc[df['hour_beginning']=="2019-11-03 01:00:00"]
```
:::

::: {.cell .markdown}


#### Handle missing values

:::


::: {.cell .markdown}


Now that we have evaluated the "completeness" of our data, we have to decide what to do about missing values. 


Some machine learning models cannot tolerate data with missing values. Depending on what *type* of data is missing and *why* it is missing, we can

* drop rows with missing values from the dataset
* fill in ("impute") the missing values with some value: a 0, the mode of that column, the median of that column, or forward/back fill data from the nearest row that is not missing


:::


::: {.cell .markdown}


For this data, let's try the forward/back fill method. This makes some sense because the data has a logical order in time, and the missing value - weather - changes relatively slowly with respect to time. We can expect that the weather at any given hour is probably similar to the weather in the previous (or next) hour.

For this to work, we'll first have to sort the data by time. (Note that the data was not sorted originally.)

:::


::: {.cell .code}

```python
df = df.sort_values(by='hour_beginning')
df.head()
```
:::


::: {.cell .markdown}


We can also "reset" the index now, so that if we ask for `df[0]` we'll get the first row in time, and so on.
:::


::: {.cell .code}

```python
df.reset_index(drop=True)
df.head()
```
:::




::: {.cell .markdown}

Now we can fill in missing data using the `fillna` function ([reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)). We will fill the missing weather data using the "forward fill" method, which caries the last valid observation forward to fill in NAs.


:::


::: {.cell .code}
```python
df['temperature'] = df['temperature'].fillna(method="ffill")
df['precipitation'] = df['precipitation'].fillna(method="ffill")
df['weather_summary'] = df['weather_summary'].fillna(method="ffill")

```
:::


::: {.cell .markdown}

Having imputed missing vaules in the weather-related columns, we can count the NAs again and find that there are only missing values in the `events` column. 

:::

::: {.cell .code}
```python
df.isnull().sum()
```
:::


::: {.cell .markdown}


#### Validating expectations

:::

::: {.cell .markdown}

Now that we have some idea of the completeness of the data, let's look at whether the data values are consistent with our expectations.


To start, let's look at summary statistics.  The "five number summary" - extremes (min and max), median, and quartiles -can help us gain a better understanding of numeric fields in the data, and see whether they have reasonable values. We can use the `describe` function in `pandas` to compute this summary.

:::



::: {.cell .code}
```python
df.describe()
```
:::

::: {.cell .markdown}

We can only compute those summary statistics for numerical variables. For categorical variables, we can use `value_counts()` to get frequency of each value.

For example, let's see how often each `weather` condition occurs, and whether it is reasonable for NYC: 

:::

::: {.cell .code}
```python
df.weather_summary.value_counts()
```
:::


::: {.cell .markdown}

It's also useful to verify expected relationships. 


For example, we expect to see precipitation when the weather is rainy. We can use `groupby` in `pandas` to capture the effect between a categorical variable (`weather_summary`) and a numerical one, `precipitation`:

:::

::: {.cell .code}
```python
df.groupby('weather_summary')['precipitation'].describe()
```
:::


::: {.cell .markdown}

Make special note of the `count` column, which shows us the prevalence of different weather conditions in this dataset. There are some weather conditions for which we have very few examples.

:::

::: {.cell .markdown}

Similarly, we can validate our expectation of hotter weather in the summer months:
:::

::: {.cell .code}
```python
df.groupby('month')['temperature'].describe()
```
:::


::: {.cell .markdown}

as well as during the middle of the day:

:::

::: {.cell .code}
```python
df.groupby('hour')['temperature'].describe()
```
:::

::: {.cell .markdown}

#### Create a pairplot

::::

::: {.cell .markdown}

For tabular data with multiple numeric features, it is often useful to create a *pairplot*. A pairplot shows pairwise relationships between all numerical variables. It is a useful way to identify variables that have a relationship.


:::

::: {.cell .markdown}

We can create a "default" pairplot with

:::

::: {.cell .code}
```python
sns.pairplot(df)
```
:::

::: {.cell .markdown}

Here, each pane shows one numerical variable on the x-axis and another numerical variable on the y-axis, so that we can see if a relationship exists between them. The panes along the diagonal shows the empirical distribution of values for each feature in this data.

:::



::: {.cell .markdown}

But, it is difficult to see anything useful because there is so much going on in this plot. We can improve things somewhat by:

* specifying only the variables we want to include, and exluding variables that don't contain useful information, such as `lat` and `long`, and
* making the points on the plot smaller and partially transparent, to help with the overplotting.


We'll also change the histograms on the diagonal, which show the frequency of values for each variable, into a density plot which shows the same information in a more useful format.

:::


::: {.cell .code}
```python
sns.pairplot(df, 
             vars=['Pedestrians', 'temperature', 'precipitation', 'hour', 'month'],
             diag_kind = 'kde',
             plot_kws={'alpha':0.5, 'size': 0.1})
```
:::


::: {.cell .markdown}

This plot validates the relationship between `temperature` and `hour`, and between `temperature` and `month`. However, we can also use this plot to identify useful features - features that appear to be related to the `target` variable.



:::

::: {.cell .markdown}


### Explore relationships and identify target variable and features


:::

::: {.cell .markdown}

Finally, since our goal is to train a machine learning model, we want to identify:

* an appropriate target variable - something on which to train our model. (Either a direct target variable, or a proxy.)
* features that are predictive - if there is any noticeable relationship between the target variable and any other variable, this is likely to be a useful feature.
* features that are correlated with one another - if two features are highly correlated, this presents some difficulty to certain types of models, so we'll want to know about it.

:::

::: {.cell .markdown}

The `Pedestrians` variable is the obvious target variable for this learning problem: it's exactly the quantity we want to predict.

:::



::: {.cell .markdown}

To identify potential predictive features among the numeric variables in the data, we can use the pairplot. Look at the row of the pairplot in which `Pedestrians` is on the vertical axis, and each of the other variables in turn is on the horizontal axis. Which of these seem to show a relationship? (Note: the relationship does not necessarily need to be a linear relationship.)

:::


::: {.cell .markdown}

We will also want to evaluate the categorical variables. For example, to look for a relationship between day of the week and pedestrian volume, we can group by `day_name`, then call the `describe` function on the `Pedestrians` column:

:::

::: {.cell .code}
```python
df.groupby('day_name')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

Similarly, we can see the effect of weather:

:::

::: {.cell .code}
```python
df.groupby('weather_summary')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

And the effect of various holidays:

:::

::: {.cell .code}
```python
df.groupby('events')['Pedestrians'].describe()
```
:::



::: {.cell .markdown}

Now armed with information about these relationships, we can identify good candidate features for a machine learning model.

:::