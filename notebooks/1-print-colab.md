::: {.cell .markdown}

# Printing from Colab

_Fraida Fund_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ffund/ml-notebooks/blob/master/notebooks/1-print-colab.ipynb)

:::


::: {.cell .markdown}

To submit homework assignments, you will need to generate PDF versions of your completed Colab notebooks. 

Printing to a PDF from Colab seems easy - there's a File > Print option in the menu! However, the built-in print option won't always work well for us, because if a plot or other output happens to come out near a page break, it can get cut off.

:::

::: {.cell .markdown}

As an alternative to Colab's built-in print, you can use this notebook to generate a PDF version of any Colab notebook that is saved in your Google Drive.

:::


::: {.cell .markdown}

## Step 1: Prepare the source notebook

Make sure the notebook that you want to print is ready:

* you ran the cells in the notebook (in order!) and their output is visible in the notebook
* it is saved in your Google Drive

:::

::: {.cell .markdown}

## Step 2: Install software and libraries

In *this* notebook, run the following cell:
:::

::: {.cell .code}
``` {.python}
!apt-get update
!apt-get install texlive texlive-xetex texlive-latex-extra pandoc
!pip install pypandoc
```
:::

::: {.cell .markdown}

## Step 3: Mount your Google Drive

In *this* notebook, mount your Google Drive:

:::

::: {.cell .code}
``` {.python}
from google.colab import drive
drive.mount('/content/drive')
```
:::

::: {.cell .markdown}

## Step 4: Select notebook and convert to PDF

In *both* of the following cells, change the name "Untitled" to whatever your notebook is named. Then, run the cells.

:::

::: {.cell .code}
``` {.python}
!jupyter nbconvert --output-dir='/content' --to latex  '/content/drive/My Drive/Colab Notebooks/Untitled.ipynb'
```
:::

::: {.cell .code}
``` {.python}
!buf_size=1000000 xelatex --interaction=nonstopmode 'Untitled.tex'
```
:::


::: {.cell .markdown}

## Step 5: Download PDF

Finally, open the Colab file browser, locate your new PDF, and download it. Review the PDF and make sure it looks good before you submit!

:::