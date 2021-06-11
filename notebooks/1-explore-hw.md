---
title:  'Exploratory data analysis'
author: 'Fraida Fund'
---


::: {.cell .markdown}

# Assignment: Exploratory data analysis


**TODO**: Edit this cell to fill in your NYU Net ID and your name:

* **Net ID**:
* **Name**:

:::


::: {.cell .markdown}


## Introduction

In this assignment, we will practice using exploratory data analysis on Google's COVID-19 Community Mobility data.

This data was collected from Google Maps users around the world over the last few months - including you, *if* you have Google Maps on your phone and have turned on the Location History setting. It combines location history from a large number of users to capture the overall increase or decrease in time spent in places such as: retail and recreation facilities, groceries and pharmacies, parks, transit stations, workplaces, and residences.

The data shows how users' mobility patterns - what types of places they spend time in - varied over the course of the COVID-19 pandemic.

As you work through this notebook, you will see that some text and code cells are marked with a "TODO" at the top. You'll have to edit these cells to fill in the code or answer the questions as indicated.

When you are finished, make sure you have run all of the cells in the notebook (in order), and then create a PDF from it. Submit the PDF on Gradescope.

**Important note**: You won't necessarily have learned or seen in advance how to use all the Python commands and library functions you need to complete this assignment. That's OK. Part of the learning objective here is to practice finding and applying that kind of new information as you go! Use the library documentation, search the Internet, or ask questions on Ed if you need any help.

:::


::: {.cell .markdown}

## Learn about the data

First, it is worthwhile to learn more about the data: how it is collected, what is included, how Google gets consent to collect this data, and how user privacy is protected. Google provides several resources for learning about the data:

* [Blog post](https://www.blog.google/technology/health/covid-19-community-mobility-reports?hl=en)
* [About this data](https://www.google.com/covid19/mobility/data_documentation.html?hl=en#about-this-data)
* [Understand the data](https://support.google.com/covid19-mobility/answer/9825414?hl=en&ref_topic=9822927)

:::



::: {.cell .markdown}

## Read in data

Now you are ready to read the data into your notebook.

Visit Google's web page for the [COVID-19 Community Mobility](https://www.google.com/covid19/mobility/) project to get the URL for the data. 

(Specific instructions will depend on your browser and operating system, but on my laptop, I can get the URL by right-clicking on the button that says "Download global CSV" and choosing "Copy Link Address".)

Then, in the following cells, use that URL to read the data into a pandas Data Frame called `df`. (You can follow the example in the "Exploratory data analysis" notebook from this week's lesson.)

:::

::: {.cell .code}
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
:::


::: {.cell .code}
```python
# TODO Q1
# url = ...
# df = ...
```
:::

::: {.cell .markdown}

Use the `info()` and `head()` functions to show some basic information about the data and to look at the first few samples.


:::


::: {.cell .code}
```python
# TODO Q2
# use info()
```
:::

::: {.cell .code}
```python
# TODO Q3
# use head()
```
:::

::: {.cell .markdown}

## Basic data manipulations

:::

::: {.cell .markdown}

The data includes a date field, but it may have been read in as a string, rather than as a `datetime`. If that's the case, use `to_datetime()` to convert the field into a datetime format. (You can follow the example in the "Exploratory data analysis" notebook from this week's lesson.) 

Then, use `info()` again to make sure your change was applied. Note the difference in the output, relative to the cell above.

:::


::: {.cell .code}
```python
# TODO Q4
# df['date'] = ...

```
:::



::: {.cell .markdown}

Next, you are going to extract the subset of data for the U.S. state of your choice. You can choose any location *except* New York.


The data is reported for different regions, with different levels of granularity available. This is best explained by example:

Suppose I want the overall trend from the entire U.S. I would use the subset of data where `country_region` is equal to "United States" and `sub_region_1` is null:

```
df_subset = df[(df['country_region'].eq("United States")) & (df['sub_region_1'].isnull())]
```

Suppose I want the overall trend from the entire state of New York: I would use the subset of data where `country_region` is equal to "United States", `sub_region_1` is equal to "New York", and `sub_region_2` is null:

```
df_subset = df[(df['country_region'].eq("United States")) & (df['sub_region_1'].eq("New York")) & (df['sub_region_2'].isnull())]
```

Suppose I want the overall trend from Brooklyn, New York (Kings County): I would use the subset of data where `country_region` is equal to "United States", `sub_region_1` is equal to "New York", and `sub_region_2` is equal to "Kings County":

```
df_subset = df[(df['country_region'].eq("United States")) & (df['sub_region_1'].eq("New York")) & (df['sub_region_2'].eq("Kings County"))]
```

In the following cell(s), fill in the code to create a data frame `df_subset` with data from a single U.S. state.


:::



::: {.cell .code}
```python
# TODO Q5
# df_subset =
```
:::


::: {.cell .markdown}

Is the data complete, or is some data not available for the location you have chosen? In the following cell, write code to check for missing data in the `...percent_change_from_baseline` fields. 

Also check whether there are any missing rows of data. What date range is represented in this data? Is every day within that range included in the data?

:::

::: {.cell .code}
```python
# TODO Q6
# df_subset
```
:::


::: {.cell .markdown}

**TODO** Q7:  Edit this cell to answer the following question: Is the data complete, or is some relevant data missing? Why would some locations only have partial data available (missing some `...percent_change_from_baseline` fields for some dates)? (Even if, for the U.S. state you have chosen, the data is complete, explain why some data may be missing for other regions.)

**Include a short quote from the material you read in the "Learn about the data" section to answer this question. Indicate that it is a quote using quotation marks or a block quote, and cite the source, including a URL.**

:::


::: {.cell .markdown}

To track trends in cases and vaccinations alongside mobility trends, we can also read in data from several other sources. For example,

* Our World in Data distributes data about COVID-19 vaccination status over time for U.S. states in their [Github repository](https://github.com/owid/covid-19-data).
* The New York Times distributes data about COVID-19 cumulative cases over time for U.S. states in their [Github repository](https://github.com/nytimes/covid-19-data).


You can choose whether to look at vaccination trends or case trends for the U.S. state you have selected. Use one of the following cells to read in the data, convert the `date` field to a `datetime`, and get the subset of the data that applies to the specific U.S. state for which you are exploring mobility data. 

Then, use `pandas` functions to check your new data frame and look at the first few rows of data.

:::


::: {.cell .code}
``` {.python}
# TODO Q8 - Vaccinations option

url_vax = 'https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/us_state_vaccinations.csv'
# df_vax = ...
# df_vax['date'] = ...
# df_vax_subset = ...
# check the data frame and look at a few rows
```
:::

::: {.cell .code}
``` {.python}
# TODO Q8 - Cases option

url_cases = 'https://github.com/nytimes/covid-19-data/raw/master/us-states.csv'
# df_cases = ...
# df_cases['date'] = ...
# df_cases_subset = ...
# check the data frame and look at a few rows
```
:::


::: {.cell .markdown}

## Visualize data

Finally, we are going to visualize the changes in human mobility over this time, for the location you have chosen, alongside either vaccination trends or cases trends.

In the following cell, create a figure with seven subplots, arranged vertically. (You can refer to the example in the "Python + numpy" notebook from this week's lesson.) On the horizontal axis, put the `days_since...` array you computed in the previous cell. On the vertical axes, show (as a line):

* `retail_and_recreation_percent_change_from_baseline` in the top subplot
* `grocery_and_pharmacy_percent_change_from_baseline` in the next subplot
* `parks_percent_change_from_baseline` in the next subplot
* `transit_stations_percent_change_from_baseline` in the next subplot
* `workplaces_percent_change_from_baseline` in the next subplot
* `residential_percent_change_from_baseline` in the next subplot
* either COVID-19 cases or vaccinations in the bottom subplot

Make sure to clearly label each axis. Use `matplotlib` library documentation to adjust your figures and make your plot look nice!

:::

::: {.cell .code}
```python
# TODO Q9
# create visualization
```
:::



::: {.cell .markdown}

**TODO** Q10: Answer the following questions: 

* Do the results seem to satisfy "common sense"? 
* Make sure to explain any trends, patterns, or notable anomalies observed in your mobility data. 
* Which trends, patterns, or notable anomalies in the mobility data are likely related to COVID-19 cases, non-pharmaceutical interventions such as stay-at-home orders, or vaccinations? 
* Which trends, patterns, or notable anomalies in the mobility data are likely related to other factors?
* Cite specific evidence from your plot to support your answer. 

**TODO** Q11: In the [Calibrate Region](https://support.google.com/covid19-mobility/checklist/9834261?hl=en&ref_topic=9822927) checklist, Google suggests a number of reasons why their mobility data might *not* be useful for understanding the effect of COVID-19-related interventions, or why the data might be misleading. 

* For the U.S. state you have chosen, briefly answer *all* of the questions in that checklist, and explain how your answer affects the validity of the data. 

* Based on your answers, do you think there are any serious problems associated with using this data for understanding user mobility changes due to COVID-19?



:::


