---
title:  'Homework: Classifying your own handwritten digit'
author: 'Fraida Fund'
jupyter:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
## Homework problem: Logistic regression for classification of handwritten digits
:::

::: {.cell .markdown }
For this homework problem, you will create your own test image for the
logistic regression classifier that we trained in a demo notebook this
week.
:::

::: {.cell .markdown }
#### Train your classifier

First, we'l repeat the steps from the demo notebook to train a
logistic regression for classification of handwritten digits. This code
is provided for you.

(It is copied from the demo notebook exactly, with one exception: we use
a larger subset of the data for training than in the demo notebook, so
this fitted model will have better accuracy.)
:::

::: {.cell .code }
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
:::

::: {.cell .code }
```python
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
```
:::

::: {.cell .code }
```python
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)
```
:::

::: {.cell .code }
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9,
                                     train_size=0.7, test_size=0.3)
```
:::

::: {.cell .code }
```python
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
:::

::: {.cell .code }
```python
clf = LogisticRegression(penalty='none', 
                         tol=0.01, solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
```
:::

::: {.cell .code }
```python
accuracy = clf.score(X_test_scaled, y_test)
```
:::

::: {.cell .code }
```python
print(accuracy)
```
:::

::: {.cell .markdown }
#### Create a test image

On a plain white piece of paper, in a black or other dark-colored pen or
pencil, write a digit of your choice from 0 to 9. Take a photo of your
handwritten digit.

Edit your photo (crop, rotate as needed), using a photo editor of your
choice (I used Google Photos), so that your photo is approximately
square, and includes only the digit and the white background. Leave a
small margin around the edge of the writing, but not too much. Your
edited photo should look similar to the MNIST images in the demo
notebook.

For example:

`<img src="https://i.ibb.co/RzLP8nm/20200710-115731.jpg" alt="A handwritten '8'" width=200/>`{=html}
:::

::: {.cell .markdown }
#### Upload your image to Colab

Run the following cell. Click "Choose files", and upload the photo of
your handwritten digit.
:::

::: {.cell .code }
```python
from google.colab import files
 
uploaded = files.upload()
 
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```
:::

::: {.cell .markdown }
On the left side of the Colab window, you will see a small file folder
icon, which allows you to explore the filesystem of your Colab
workspace. If you click on this icon, you should see that your file has
been uploaded to your Colab workspace. (You may need to use the
"Refresh" button in the file browser in order to see the file.) Make a
note of the file name.
:::

::: {.cell .markdown }
#### Visualize the image

After uploading your image, run this cell, but *replace the filename*
with the name of the file you have just uploaded to Colab. You shold see
your image appear in the cell output.
:::

::: {.cell .code }
```python
from PIL import Image
 
filename = '2021-07-01_14-03.png'
 
image = Image.open(filename)
p = plt.imshow(np.asarray(image), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image).shape))
```
:::

::: {.cell .markdown }
For example:

`<img src="https://i.ibb.co/jy2Qt6Z/image.png" alt="A handwritten '8' after uploading to Colab" width=200/>`{=html}
:::

::: {.cell .markdown }
#### Pre-process the image

The images in MNIST have been pre-processed - they are converted to
grayscale, and centered in a 28x28 image by computing the center of mass
of the pixels, and then translating and scaling the image so as to
position this point at the center of the 28x28 field.

You have already done some manual pre-processing, by cropping your image
before uploading. But you may have noticed from the `shape` output that
your image resolution is much larger than 28x28, and you probably had
three color channels (red, green, and blue).

Use the code in the following cells to pre-process your image into a
28x28 image with one color channel (grayscale). You may have to manually
tune the contrast for best results, by changing the `pixel_filter`
value. You will want the background to be as close to pure black as
possible, without affecting the legibility of the handwritten digit.

(We won't bother with centering the image, but that would probably
improve the prediction performance quite a lot!)
:::

::: {.cell .code }
```python
# convert to grayscale image - 'L' format means each pixel is 
# represented by a single value from 0 to 255
image_bw = image.convert('L')
p = plt.imshow(np.asarray(image_bw), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw).shape))
```
:::

::: {.cell .code }
```python
# resize image 
image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
p = plt.imshow(np.asarray(image_bw_resized), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw_resized).shape))
```
:::

::: {.cell .code }
```python
# invert image, to match training data
import PIL.ImageOps    

image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
p = plt.imshow(np.asarray(image_bw_resized_inverted), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw_resized_inverted).shape))
```
:::

::: {.cell .code }
```python
# adjust contrast and scale
pixel_filter = 20 # value from 0 to 100 - may need to adjust this manually
min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
max_pixel = np.max(image_bw_resized_inverted_scaled)
image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
p = plt.imshow(np.asarray(image_bw_resized_inverted_scaled), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(np.asarray(image_bw_resized_inverted_scaled).shape))
```
:::

::: {.cell .code }
```python
# finally, reshape to (1, 784) - 1 sample, 784 features
test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(test_sample.shape))
```
:::

::: {.cell .markdown }
Make sure the `shape` of your test sample is `(1,784)` (one sample, 784
features).
:::

::: {.cell .markdown }
#### Visualize the pre-processed image

Run the following code to visualize your pre-processed image.
:::

::: {.cell .code }
```python
p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,);
p = plt.title('Shape: ' + str(test_sample.shape))
```
:::

::: {.cell .markdown }
For example:

`<img src="https://i.ibb.co/0rD9Z75/image.png" alt="A handwritten '8' after pre-processing" width=200/>`{=html}
:::

::: {.cell .markdown }
#### Use your fitted logistic regression

Now that you have processed your test image, let us see whether it is
classified correctly by the logistic regression.
:::

::: {.cell .markdown }
Run the following cell. This will use your fitted logistic regression to
predict conditional probabilities per class for this test sample, and
plot them.
:::

::: {.cell .code }
```python
test_probs = clf.predict_proba(test_sample)

sns.barplot(x=np.arange(0,10), y=test_probs.squeeze());
plt.ylabel("Probability");
plt.xlabel("Class");
```
:::

::: {.cell .markdown }
For example:

`<img src="https://i.ibb.co/80TzWQv/image.png" alt="Probabilities" />`{=html}
:::

::: {.cell .markdown }
Also run this cell, to show the predicted label for your test sample:
:::

::: {.cell .code }
```python
```
:::

::: {.cell .code }
```python
test_pred = clf.predict(test_sample)
print("Predicted class is: ", test_pred)
```
:::

::: {.cell .markdown }
#### Explain the model prediction

Even if the fitted model correctly labeled your handwritten digit, it
may have estimated a moderately high probability for some of the other
labels. To understand why, it is useful to visualize

$$\langle w_k, x\rangle$$

for each class $k$.

Add a cell with the following code, and run it. This will plot:

-   on the top row, the coefficient vector for each class,
-   on the bottom row, each pixel in your test image, multiplied by the
    associated coefficient for that class.
:::

::: {.cell .code }
```python
scale = np.max(np.abs(clf.coef_))

p = plt.figure(figsize=(25, 5));

for i in range(nclasses):
    p = plt.subplot(2, nclasses, i + 1)
    p = plt.imshow(clf.coef_[i].reshape(28, 28),
                  cmap=plt.cm.RdBu, vmin=-scale, vmax=scale);
    p = plt.title('Class %i' % i);
    p = plt.axis('off')

for i in range(nclasses):
    p = plt.subplot(2, nclasses, nclasses + i + 1)
    p = plt.imshow(test_sample.reshape(28, 28)*clf.coef_[i].reshape(28, 28),
                  cmap=plt.cm.RdBu, vmin=-scale/2, vmax=scale/2);
    # note: you can adjust the scaling factor if necessary,
    # to make the visualization easier to understand
    p = plt.axis('off')
```
:::

::: {.cell .markdown }
For example:

`<img src="https://i.ibb.co/MGLkf0T/image.png" alt="A handwritten '8' after pre-processing"/>`{=html}
:::

::: {.cell .markdown }
In the images in the bottom row,

-   a blue pixel (and especially a dark blue pixel) means that your test
    image had writing in the part of the image that is positively
    associated with belonging to the class, and
-   a red pixel (and especially a dark red pixel) means that your test
    image had writing in the part of the image that is negatively
    associated with belonging to the class.
:::

::: {.cell .markdown }
### Exploring the model error

The image above should give you an idea of why your digit was classified
correctly or incorrectly, and should help you understand when and why
the model misclassifies some samples.

-   if your image *was* classified correctly: draw a *slightly* modified
    version of the same digit, that you believe will be classified
    *incorrectly*. Run this second image through the steps above, and
    confirm your intuition.
-   if your image *was not* classified correctly: draw a *slightly*
    modified version of the same digit, that you believe will be
    classified *correctly*. Run this second image through the steps
    above, and confirm your intuition.

(Your second image should still be approximately square, include only
the digit and the white background, and have a small margin around the
edge of the writing, i.e. it should also "look like" the MNIST
samples.)
:::

::: {.cell .markdown }
### What to submit

Don't submit the entire notebook. Instead, submit only the following
items (for *your two handwritten digit samples*, not my example):

-   The visualization of your test image before pre-processing.
-   The visualization of your test image after pre-processing.
-   The bar plot showing the conditional probabilities per class for
    your test image.
-   The predicted class label for your test image.
-   The figure from the "Explain the model prediction" section.
-   **In your own words**, list the classes for which the logistic
    regression predicted a high or moderately high probability. Using
    the figure from the "explain the model prediction" section,
    explain *why* the logistic regression estimates that these classes
    are very likely or moderately likely.
-   Explain: how did you know what changes to make to your original
    drawing to create a modified version that would get a different
    predicted class label?
:::
