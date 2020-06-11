---
title:  'Exploring a candidate data set'
---


::: {.cell .markdown}
# Exploring a candidate data set

_Fraida Fund_


:::


::: {.cell .markdown}


## Introduction

In this notebook, we will consider several machine learning tasks (satirical headline classification, chest X-ray classification, and candidate data sets for them. We will explore the following questions:

* Do these data sets seem appropriate for the task? 
* Are there any important limitations of the datasets, or problems that need addressing before we use them to train a machine learning model?  

In fact, each of these datasets has a significant problem that - if not detected early on - would be a "Garbage In, Garbage Out" situation. See if you can identify the problem with each dataset!

To get you started, I included some code to show you how to read in the data. You can add additional code and text cells to explore the data. If you find something interesting, share it on Piazza!

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

You are developing an app for NYC taxi drivers that will predict what the typical tip would be for a given fare. You consider using data collected by the NYC Taxi and Limousine Commission on taxi trips. The links are for 2019 data, but previous years are also available. [Data link for yellow (Manhattan) taxi trips](https://data.cityofnewyork.us/Transportation/2019-Yellow-Taxi-Trip-Data/2upf-qytp) and [data link for green (non-Manhattan) taxi trips](https://data.cityofnewyork.us/Transportation/2019-Green-Taxi-Trip-Data/q5mz-t52e)

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

Is the cell above taking a long time to run? That's because this data set is very large, and the server from which it is retrieved is not very fast. Since we don't need to explore the whole dataset, necessarily, we can interrupt the partial download by clicking on the square icon to the left of the cell that is running.

Then, we can read in just 10,000 rows of data.
:::



::: {.cell .code}
```python
df_taxi = pd.read_csv('2019-Green-Taxi-Trip-Data.csv', nrows=10000)   
df_taxi.head()
```
:::




::: {.cell .markdown}

## Highway traffic prediction

:::



::: {.cell .markdown}

### Scenario

You are working for the state of New York to develop a traffic prediction model for the NYS Thruway. The following Thruway data is available: Number and types of vehicles that entered from each entry point on the Thruway, along with their exit points, at 15 minute intervals. The link points to the most recent week's worth of available data, but this data is available through 2014. [Link to NYS Thruway data](https://data.ny.gov/Transportation/NYS-Thruway-Origin-and-Destination-Points-for-All-/4dbf-24u2) 

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

## Satirical headline classification


:::


::: {.cell .markdown}

### Scenario

You are hired by a major social media platform to develop a machine learning model that will be used to clearly mark *satirical news articles* when they are shared on social media. 
You consider using this dataset of 9,000 headlines from [The Onion](https://www.theonion.com/) and 15,000 headlines from [Not The Onion on Reddit](https://www.reddit.com/r/nottheonion/). [Link to OnionOrNot data](https://github.com/lukefeilberg/onion)

:::


::: {.cell .markdown}

### Read in data

This time, we'll retrieve the data from Github.

:::


::: {.cell .code}
```python
!git clone https://github.com/lukefeilberg/onion.git
%cd onion
```
:::

::: {.cell .code}
```python
df_headline = pd.read_csv("OnionOrNot.csv")
```
:::


::: {.cell .markdown}

## Offensive post classification

:::


::: {.cell .markdown}

### Scenario

The social media platform was so impressed with your work on detection of satirical headlines, that they asked you to work on a model to identify posts using offensive language. 
As training data, they hand you 80,000 tweets, labeled as either "hateful", "abusive", "spam", or "none", by majority vote of five people. [Link to abusive tweets data](https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN)


:::


::: {.cell .markdown}

### Read in data


This time, we'll read in data to Colab by downloading it to our own computer from the link above, then uploading it to Colab.

Use the interactive file upload form below to upload the `hatespeechtwitter.csv` file.


:::


::: {.cell .code}
```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```
:::


::: {.cell .code}
```python
df_tweets = pd.read_csv('hatespeechtwitter.csv')
df_tweets
```
:::

::: {.cell .markdown}


## Chest X-ray classification

:::


::: {.cell .markdown}

### Scenario

You are working for a large hospital system to develop a machine learning model that, given a chest X-ray, should identify those that likely have COVID-19 so that they can take proper precautions against the spread of infection within the hospital. 
You consider using two datasets together: one with several hundred images of chest X-rays of likely COVID-19 patients, and a pre-COVID dataset of chest X-ray images. [Link to COVID-19 chest X-ray data](https://github.com/ieee8023/covid-chestxray-dataset), [Link to pre-COVID chest X-ray data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)

:::

::: {.cell .markdown}

### Read in data

First, we will download the RSNA data from the [RSNA website](https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018).

Then, we'll also retrieve the COVID-19 data from Github.

:::


::: {.cell .markdown}

#### RSNA data

:::



::: {.cell .code}
```python
!wget https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip -O pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip
```
:::




::: {.cell .code}
```python
!mkdir rsna
!unzip -j -d rsna/ pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip
```
:::


::: {.cell .markdown}

Now, we'll make a list of all the image files:

:::



::: {.cell .code}
```python
import glob
rsna_images = glob.glob("rsna/*.dcm")
len(rsna_images)
```
:::



::: {.cell .code}
```python
rsna_images[:5]
```
:::


::: {.cell .markdown}

These images are in DICOM format, a medical imaging file format. We need to install an extra library to read them in: 

:::


::: {.cell .code}
```python
!pip install pydicom
```
:::




::: {.cell .code}
```python
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
```
:::


::: {.cell .markdown}

Now, we can read in one file from the list:

:::


::: {.cell .code}
```python
ref_xray = dicom.read_file(rsna_images[0])
ref_xray
```
:::




::: {.cell .code}
```python
dir(ref_xray)
```
:::

::: {.cell .markdown}

We'll find out the dimensions of the image, then represent it as an array of pixels, and plot it:

:::




::: {.cell .code}
```python
pixel_dims = (int(ref_xray.Rows), int(ref_xray.Columns))
pixel_dims
```
:::


::: {.cell .code}
```python
ref_xray.pixel_array.shape
```
:::




::: {.cell .code}
```python
print(ref_xray.pixel_array)
```
:::




::: {.cell .code}
```python
plt.imshow(ref_xray.pixel_array, cmap='bone')
```
:::



::: {.cell .markdown}

#### COVID-19 data

:::



::: {.cell .code}
```python
!git clone https://github.com/ieee8023/covid-chestxray-dataset
```
:::




::: {.cell .code}
```python
covid_metadata = pd.read_csv('covid-chestxray-dataset/metadata.csv')
covid_metadata.info()
```
:::




::: {.cell .code}
```python
covid_metadata.head()
```
:::



::: {.cell .code}
```python
covid_metadata.modality.value_counts()
```
:::




::: {.cell .code}
```python
covid_metadata.finding.value_counts()
```
:::


::: {.cell .markdown}

We're going to pull out a subset of the data that (1) is a chest X-ray, not CT, and (2) has a positive COVID-19 finding, 
:::


::: {.cell .code}
```python
covid_xray_metadata = covid_metadata[(covid_metadata["modality"] == "X-ray") & (covid_metadata["finding"] == "COVID-19")]
covid_xray_metadata.info()
```
:::


::: {.cell .markdown}

Make a list of image files:

:::


::: {.cell .code}
```python
covid_images = 'covid-chestxray-dataset/images/' +  covid_xray_metadata['filename']
len(covid_images)
```
:::




::: {.cell .code}
```python
covid_images
```
:::

::: {.cell .markdown}

We'll use the PIL library to read in JPG and PNG files, and plot one:

:::




::: {.cell .code}
```python
from PIL import Image
```
:::



::: {.cell .code}
```python
image = Image.open(covid_images[0])
image_bw = image.convert('L') #  L is 8-bit pixels, black and white
```
:::




::: {.cell .code}
```python
image_data = np.asarray(image_bw)
image_data.shape
```
:::




::: {.cell .code}
```python
plt.imshow(image_bw, cmap='bone')
```
:::


::: {.cell .markdown}

#### Plot samples of each

:::


::: {.cell .code}
```python
num_classes = 2
samples_per_class = 10
figure = plt.figure(figsize=(samples_per_class*3, num_classes*3))

# plot RSNA samples
rsna_samples = np.random.choice(rsna_images, samples_per_class, replace=False)
for i, sample in enumerate(rsna_samples):
    plt_idx = i + 1
    plt.subplot(num_classes, samples_per_class, plt_idx)
    sample_img = dicom.read_file(sample).pixel_array
    plt.imshow(sample_img, cmap='bone')
    plt.axis('off')
    plt.title("Non-COVID")


# plot COVID samples
covid_samples = np.random.choice(covid_images, samples_per_class, replace=False)
for i, sample in enumerate(covid_samples):
    plt_idx = samples_per_class + i + 1
    plt.subplot(num_classes, samples_per_class, plt_idx)
    sample_img = Image.open(sample)
    sample_image_bw = sample_img.convert('L')
    plt.imshow(sample_image_bw, cmap='bone')
    plt.axis('off')
    plt.title("COVID-19")

plt.show()
```
:::
 