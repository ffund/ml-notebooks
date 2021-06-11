---
title:  'Data detective challenge'
author: 'Fraida Fund'
---


::: {.cell .markdown}

# Data detective challenge!

_Fraida Fund_


:::


::: {.cell .markdown}


## Introduction

In this notebook, we will consider several machine learning tasks, and candidate data sets for them. We will explore the following questions:

* Do these data sets seem appropriate for the task? 
* Are there any important limitations of the datasets, or problems that need to be addressed before we use them to train a machine learning model?  

In fact, each of these datasets has a significant problem that - if not detected early on - would create a "Garbage In, Garbage Out" situation. See if you can identify the problem with each dataset!

To get you started, I included some code to show you how to read in the data. You can add additional code and text cells to explore the data. 

Your work on this challenge won't be submitted or graded. If you think you found the problem with a dataset, share your findings with the class by posting on Ed! (In your post, show evidence from your exploratory data analysis to support your claims.)


:::



::: {.cell .code}
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
:::


::: {.cell .markdown}

## Taxi tip prediction

:::


::: {.cell .markdown}

### Scenario

You are developing an app for NYC taxi drivers that will predict what the typical tip would be for a given fare. 

You consider using data collected by the NYC Taxi and Limousine Commission on taxi trips. These links are for 2019 data (2020 was probably an atypical year, so we won't use that). Previous years are also available. 

* [Data link for yellow (Manhattan) taxi trips](https://data.cityofnewyork.us/Transportation/2019-Yellow-Taxi-Trip-Data/2upf-qytp) 
* [Data link for green (non-Manhattan) taxi trips](https://data.cityofnewyork.us/Transportation/2019-Green-Taxi-Trip-Data/q5mz-t52e)

:::

::: {.cell .markdown}

### Read in data

We'll start by reading in the 2019 Green Taxi trip data. It's a large file and takes a long time to download, so we may interrupt the download in middle (using the Runtime menu in Colab) and just work with the partial data.


In the next couple of cells, `wget` and `wc` are not Python code - they're Linux commands. We can run some basic Linux commands inside our Colab runtime, and it's often helpful to do so. For example, we may use Linux commands to install extra software libraries that are not pre-installed in our runtime, clone a source code repository from Github, or download data from the Internet.

:::


::: {.cell .code}
```python
!wget "https://data.cityofnewyork.us/api/views/q5mz-t52e/rows.csv?accessType=DOWNLOAD" -O 2019-Green-Taxi-Trip-Data.csv
```
:::

::: {.cell .markdown}

Is the cell above taking a long time to run? That's because this data set is very large, and the server from which it is retrieved is not very fast. Since we don't need to explore the whole dataset, necessarily, we can interrupt the partial download by using the Runtime > Interrupt Execution menu option.

Then, we can read in just 10,000 rows of data.
:::



::: {.cell .code}
```python
df_taxi = pd.read_csv('2019-Green-Taxi-Trip-Data.csv', nrows=10000)   
df_taxi.head()
```
:::

::: {.cell .markdown}

Use additional cells as needed to explore this data. Answer the following questions:

* How is the data collected? Is it automatic, or is there human involvement?
* What variable should be the *target variable* for this machine learning problem?
* What variable(s) could potentially be used as *features* to train the model?
* What are our assumptions about the features and the target variable, and the relationships between these? (For example: in NYC, what is a conventional tip amount, as a percent of the total fare? If you are not from NYC, you can find information about this online!) Are any of these assumptions violated in this data?
* Are there variables that should *not* be used as features to train the model, because of potential for data leakage?
* Are there any serious data problems that we need to correct before using the data for this purpose? Explain.

:::


::: {.cell .markdown}

## Highway traffic prediction

:::



::: {.cell .markdown}

### Scenario

You are working for the state of New York to develop a traffic prediction model for the NYS Thruway. The following Thruway data is available: Number and types of vehicles that entered from each entry point on the Thruway, along with their exit points, at 15 minute intervals. 

The link points to the most recent week's worth of available data, but this data is available through 2014. [Link to NYS Thruway data](https://data.ny.gov/Transportation/NYS-Thruway-Origin-and-Destination-Points-for-All-/4dbf-24u2) 

:::

::: {.cell .markdown}

### Read in data

:::


::: {.cell .code}
```python
url = 'https://data.ny.gov/api/views/4dbf-24u2/rows.csv?accessType=DOWNLOAD&sorting=true'
df_thruway = pd.read_csv(url)
```
:::

::: {.cell .markdown}

Use additional cells as needed to explore this data. Answer the following questions:

* How is the data collected? Is it automatic, or is there human involvement?
* What variable should be the *target variable* for this machine learning problem?
* What variable(s) could potentially be used as *features* to train the model?
* What are our assumptions about the features and the target variable, and the relationships between these? (For example: what times of day should be busy? What times of day will be less busy? What stretches of the Thruway might be especially congested - look at Google Maps?)
* Are there variables that should *not* be used as features to train the model, because of potential for data leakage?
* Are there any serious data problems that we need to correct before using the data for this purpose? Explain.

:::


::: {.cell .markdown}

## Satirical headline classification


:::


::: {.cell .markdown}

### Scenario

You are hired by a major social media platform to develop a machine learning model that will be used to clearly mark *satirical news articles* when they are shared on social media. 

You consider using this dataset of 9,000 headlines from [The Onion](https://www.theonion.com/) and 15,000 headlines from [Not The Onion on Reddit](https://www.reddit.com/r/nottheonion/). [Link to OnionOrNot data](https://github.com/lukefeilberg/onion)

([This notebook](https://github.com/lukefeilberg/onion/blob/master/Onion.ipynb) shows how the data was compiled and processed.)

:::


::: {.cell .markdown}

### Read in data

This time, we'll retrieve the data from Github.

:::


::: {.cell .code}
```python
!git clone https://github.com/lukefeilberg/onion.git
```
:::

::: {.cell .code}
```python
df_headline = pd.read_csv("onion/OnionOrNot.csv")
```
:::

::: {.cell .markdown}

::: {.cell .markdown}

Use additional cells as needed to explore this data. Answer the following questions:

* How is the data collected? Is it automatic, or is there human involvement?
* What variable should be the *target variable* for this machine learning problem?
* What variable(s) could potentially be used as *features* to train the model?
* What are our assumptions about the data?
* Are there variables that should *not* be used as features to train the model, because of potential for data leakage?
* Are there any serious data problems that we need to correct before using the data for this purpose? Explain.

:::


:::