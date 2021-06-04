::: {.cell .markdown}

# Printing from Colab

_Fraida Fund_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ffund/ml-notebooks/blob/master/notebooks/1-print-colab.ipynb)

:::


::: {.cell .markdown}

Printing from Colab seems easy - there's a File > Print option in the menu! However, the built-in print option won't always work well for us, because if a plot or other output happens to come out near a page break, it can get cut off.

For example, try running the following cell, which creates a large plot:

:::

::: {.cell .code}
``` {.python}
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 3 * np.pi, 0.1)

# set up figure and subplots
plt.figure(figsize=(6, 20))
plt.subplot(8, 1, 1)

# plot random data
for p in range(1,8+1):
  y = np.random.exponential(size=len(x))
  plt.subplot(8, 1, p)
  plt.scatter(x, y)
  plt.title('Plot %d/8' % p)

# adjust spacing between subplots
plt.subplots_adjust(hspace = 0.4)

# show the figure.
plt.show()
```
:::

::: {.cell .markdown}

Then, look at the preview PDF output using File > Print, and note how some of the subplots do not appear in the PDF output.

:::


::: {.cell .markdown}

As an alternative to Colab's built-in print, you can use this notebook to generate a PDF version of any Colab notebook that is saved in your Google Drive.

:::


::: {.cell .markdown}

## Step 1: Prepare the source notebook

Make sure the notebook that you want to print is ready:

* you ran the cells in the notebook (in order! and their output is visible in the notebook
* it is saved in your Google Drive

:::

::: {.cell .markdown}

## Step 2: Install software and libraries

In *this* notebook, run the following cell:
:::

::: {.cell .code}
``` {.python}
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

In *both* of the following cells, change the name "Untitled" to whatever your notebook is named. Then, run the cell.

:::

::: {.cell .code}
``` {.python}
!jupyter nbconvert --output-dir='/content' --to latex  '/content/drive/My Drive/Colab Notebooks/Untitled.ipynb'
```
:::

::: {.cell .code}
``` {.python}
!xelatex --interaction=nonstopmode Untitled.tex
```
:::


::: {.cell .markdown}

## Step 5: Download PDF

Finally, open the Colab file browser, locate your new PDF, and download it. Review the PDF and make sure it looks good before you submit!

:::