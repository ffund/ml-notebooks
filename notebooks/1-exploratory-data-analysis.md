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

* We practice using `pandas` to read in and manipulate a data set
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

1.  Learn about your data
2.  Load data and check that it is loaded correctly
3.  Visually inspect the data
4.  Compute summary statistics
5.  Explore the data further and look for potential issues

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

Support you are developing a machine learning model to predict the volume of pedestrian traffic on the Brooklyn Bridge. There is a dataset available that you think may be useful as training data: [Brooklyn Bridge Automated Pedestrian Counts dataset](https://www1.nyc.gov/html/dot/html/about/datafeeds.shtml#Pedestrians), from the NYC Department of Transportation.

We will practice applying the "recipe" for exploratory data analysis to this data.

We will use the `pandas` library in Python, which includes many powerful utilities for managing data. You can refer to the [`pandas` reference](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) for more details on the `pandas` functions used in this notebook.


:::


::: {.cell .markdown}


### Learn about your data

The first step is to learn more about the data:

* Read about *methodology* and *data codebook*
* How many rows and columns are in the data?
* What does each variable mean? What units are data recorded in?
* What variables could be used as target variable? What variables could be used as features from which to learn?
* How was data collected? Identify sampling issues, timeliness issues, fairness issues, etc.


:::


::: {.cell .markdown}




For the Brooklyn Bridge dataset, you can review the associated documentation on the NYC Data website:

* [NYC Data Website](https://data.cityofnewyork.us/Transportation/Brooklyn-Bridge-Automated-Pedestrian-Counts-Demons/6fi9-q3ta)
* [Data dictionary](https://data.cityofnewyork.us/api/views/6fi9-q3ta/files/845905ea-21d4-4ec7-958a-a1a09214513d?download=true&filename=Brooklyn_Bridge_Automated_Pedestrian_Counts_Demonstration_Project_data_dictionary.xlsx)


:::

::: {.cell .markdown}

### Load data and check that it is loaded correctly

The next step is to load the data in preparation for your exploratory data analysis.

:::

::: {.cell .markdown}

First, we will import some useful libraries:

* In Python - libraries add powerful functionality
* You can import an entire library (`import foo`) or part (`from foo import bar`)
* You can define a nickname, which you will use to call functions of these libraries (many libraries have "conventional" nicknames)

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

Our data is in CSV format, so will use the `read_csv` function in `pandas` to read in our data.  

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

We will want to verify that the data was loaded correctly. For *tabular* data, we can start by looking at a few rows of data with the `head` function. (For data that is not tabular, such as image, text, or audio data, we might start by looking at a few random samples instead.)

:::

::: {.cell .code}

```python
df.head()
```
:::


::: {.cell .markdown}

One thing to look for in the output above, that is easily missed: verify that column names and row names are loaded correctly, and that the first row of real data is actually data, and not column labels.


We should also check the shape of the data frame - the number of rows and columns. This, too, should be checked against our assumptions about the data from the NYC Data website. 

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

We have a date/time column that was read in as a string, so we can correct that now:


:::



::: {.cell .code}

```python
df['hour_beginning'] = pd.to_datetime(df['hour_beginning'])
df.info()
```
:::



::: {.cell .markdown}

And once we have done that, we can order the data frame by time:

:::



::: {.cell .code}

```python
df = df.sort_values(by='hour_beginning')
df.head()
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

For data that is recorded at regular time intervals, it is also important to know whether the data is complete, or whether there are gaps in time.  We will use some helpful `pandas` functions:

* [`pd.to_datetime`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html)
* [`pd.date_range`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

First, we will use `date_range` to get the list of hour intervals that we expect to find in the dataset. Then, we will find the difference between this list and the actual list of hour intervals in the dataset - these are missing intervals.


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
# then identify the missing hours
expected_range = pd.date_range(start = min_dt, end = max_dt, freq='H' )
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

This is also a good time to look for rows that are missing data in some columns ("NA" values), that may need to be cleaned. 

We can see the number of NAs in each column by summing up all the instances where the `isnull` function returns a True value:
:::

::: {.cell .code}
```python
df.isnull().sum()
```
:::

::: {.cell .markdown}

There are some rows of data that are missing weather, temperature, and precipitation data. We can see these rows with

:::

::: {.cell .code}
```python
df[df['temperature'].isnull()]
```
:::

::: {.cell .markdown}

pandas includes routines to fill in missing data using the `fillna` function ([reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)). We will fill these using the "forward fill" method, which caries the last valid observation forward to fill in NAs.

(Note: this makes sense only because we already sorted by date, and it's reasonable to expect adjacent hours to have similar weather!)

:::


::: {.cell .code}
```python
df['temperature'] = df['temperature'].fillna(method="ffill")
df['precipitation'] = df['precipitation'].fillna(method="ffill")
df['weather_summary'] = df['weather_summary'].fillna(method="ffill")

```
:::


::: {.cell .markdown}

Now we can count the NAs again and find that there are only missing values in the `events` column. This is the expected result, since there are many days with no event.

:::

::: {.cell .code}
```python
df.isnull().sum()
```
:::


::: {.cell .markdown}


### Visually inspect data

Now we are ready to visually inspect the data.

For tabular data, and especially tabular data with many numeric features, it is often useful to create a *pairplot*. A pairplot shows pairwise relationships between all numerical variables. It is a useful way to identify:

* features that are predictive - if there is any noticeable relationship between the target variable and any other variable.
* features that are correlated - if two features are highly correlated, we may be able to achieve equally good results just using one of them.


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

We are mainly interested in the top row of the plot, which shows how the target variable (`Pedestrians`) varies with the temperature, precipitation levels, and hour. However, it is also useful to note relationships between features. For example, there is a natural relationship between the time of data and the temperature, and between the month and the temperature.


:::


::: {.cell .markdown}

### Summary statistics

Now, we are ready to explore summary statistics.  The "five number summary" - extremes (min and max), median, and quartiles -can help us gain a better understanding of the data. We can use the `describe` function in `pandas` to compute this summary.

:::



::: {.cell .code}
```python
df.describe()
```
:::

::: {.cell .markdown}

We are especially interested in `Pedestrians`, the target variable, so we can describe that one separately:

:::

::: {.cell .code}
```python
df['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

For categorical variables, we can use `groupby` to get frequency and other useful summary statistics.

For example, we may be interested in the summary statistics for `Pedestrians` for different weather conditions: 

:::

::: {.cell .code}
```python
df.groupby('weather_summary')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

Make special note of the `count` column, which shows us the prevalence of different weather conditions in this dataset. There are some weather conditions for which we have very few examples.

Another categorical variable is `events`, which indicates whether the day is a holiday, and which holiday. Holidays have very different pedestrian traffic characteristics from other days. 

:::

::: {.cell .code}
```python
df.groupby('events')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

It can be useful to get the total pedestrian count for the day of a holiday, rather than the summary statistics for the hour-long intervals. We can use the `agg` function to compute key statistics, including summing over all the samples in the group:

:::


::: {.cell .code}

```python
df.groupby('events').agg({'Pedestrians': 'sum'})

```
:::


::: {.cell .markdown}

### Explore relationships and look for issues

Finally, let's further explore relationships between likely predictors and our target variable. We can group by `day_name`, then call the `describe` function on the `Pedestrians` column to see the effect of day of the week on traffic volume:

:::

::: {.cell .code}
```python
df.groupby('day_name')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

Similarly, we can see the effect of temperature:

:::

::: {.cell .code}
```python
df.groupby('temperature')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

And the effect of precipitation:

:::

::: {.cell .code}
```python
df.groupby('precipitation')['Pedestrians'].describe()
```
:::

::: {.cell .markdown}

We can even plot it separately, by saving it in a new data frame and plotting _that_ data frame:

:::

::: {.cell .code}
```python
df_precip = df.groupby('precipitation')['Pedestrians'].describe()
df_precip = df_precip.reset_index()
sns.scatterplot(data=df_precip, x='precipitation', y='50%')
```
:::

::: {.cell .markdown}

We see that certain weather conditions (very high temperature, heavy precipitation, fog) are extremely underrepresented in the dataset. This would be something to consider if, for example, we wanted to use this dataset to predict the effect of extreme weather on pedestrian traffic.

:::

