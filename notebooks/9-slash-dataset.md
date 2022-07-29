---
title: 'Learning the Slash dataset'
author: 'Fraida Fund'
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown }
# Demo: Convolutional neural networks on the "slash" dataset

*Fraida Fund*
:::

::: {.cell .markdown }
In this demo, we'll look at an example of a task that is difficult for
"classical" machine learning models, and difficult for fully connected
neural networks, but easy for convolutional neural networks.
:::

::: {.cell .code }
```python
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import train_test_split
from sklearn import ensemble, neighbors, linear_model, svm

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, BatchNormalization, InputLayer, AvgPool2D, MaxPool2D, GlobalAvgPool2D
import tensorflow.keras.backend as K
from keras.utils.vis_utils import plot_model
```
:::

::: {.cell .markdown }
## The slash dataset

The "slash" dataset, developed by [Sophie
Searcy](https://soph.info/slash-data), is a set of images, each of which
includes a "slash" on a background of random noise. The data is
divided into two classes according to whether the slash is downward
facing or upward facing.
:::

::: {.cell .code }
```python
def gen_example(size=20, label=0):

    max_s_pattern = int(size // 4)
    s_pattern = 4
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,
                                  size=(2, ))  # random x,y point
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] +
       s_pattern] = pattern  # set point to
    ex = ex + .5*(np.random.rand(size, size) - .5)
    np.clip(ex,0.,1., out=ex)
    return ex
```
:::

::: {.cell .code }
```python
examples = []

n_side = 30
n_ex = 500 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example(size=n_side, label=0))
    examples.append(gen_example(size=n_side, label=1))
    
y = np.array([0,1]*n_ex)
x = np.stack(examples)
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x[index,...], cmap='gray')
    plt.title(f"y = {y[index]}")
```
:::

::: {.cell .markdown }
We'l prepare training and test data in two formats:

-   "flat" for traditional ML models and fully connected neural
    networks, which don't care about the spatial arrangement of the
    features.
-   "image" for convolutional neural networks.
:::

::: {.cell .code }
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

x_train_img = x_train[...,np.newaxis]
x_test_img = x_test[...,np.newaxis]
```
:::

::: {.cell .code }
```python
print("Flat data shape:  ", x_train_flat.shape)
print("Image data shape: ", x_train_img.shape)
```
:::

::: {.cell .markdown }
The feature data is in the range 0 to 1:
:::

::: {.cell .code }
```python
x.min(), x.max()
```
:::

::: {.cell .markdown }
## Train logistic regression, random forest, KNN, SVM models
:::

::: {.cell .markdown }
Next, we'll try to train some classic ML models on this dataset.
:::

::: {.cell .code }
```python
models = {
    "Logistic\n Regression": linear_model.LogisticRegression(),
    "KNN-1": neighbors.KNeighborsClassifier(n_neighbors=1),
    "KNN-3": neighbors.KNeighborsClassifier(n_neighbors=3),
    "Random\n Forest": ensemble.RandomForestClassifier(n_estimators=100),
    "SVM -\n Linear": svm.SVC(kernel="linear"),
    "SVM -\n RBF kernel": svm.SVC(kernel="rbf")
}
```
:::

::: {.cell .code }
```python
results = []

for model_name in models.keys():    
    model = models[model_name]
    model.fit(x_train_flat, y_train)
    
    train_score = model.score(x_train_flat, y_train)
    test_score = model.score(x_test_flat, y_test)   
    
    results.append({"model": model_name, "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(10,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }
Are these the results we expected? Why or why not?
:::

::: {.cell .markdown }
Do *any* of these models do a good job of learning whether a slash is
forward-facing or backward-facing?
:::

::: {.cell .markdown }
## Train a fully connected neural network
:::

::: {.cell .code }
```python
nin = x_train_flat.shape[1]
nh1 = 64
nh2 = 64
nh3 = 64  
nout = 1 
model_fc = Sequential()
model_fc.add(Dense(units=nh1, input_shape=(nin,), activation='relu', name='hidden1'))
model_fc.add(Dense(units=nh2, input_shape=(nh1,), activation='relu', name='hidden2'))
model_fc.add(Dense(units=nh3, input_shape=(nh2,), activation='relu', name='hidden3'))
model_fc.add(Dense(units=nout, activation='sigmoid', name='output'))

model_fc.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_fc.summary()
```
:::

::: {.cell .code }
```python
hist = model_fc.fit(x_train_flat, y_train, epochs=100, 
     validation_split=0.25,  callbacks=[
        keras.callbacks.ReduceLROnPlateau(factor=.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1)
    ])
```
:::

::: {.cell .code }
```python
train_score = model_fc.evaluate(x_train_flat, y_train)[1]
test_score = model_fc.evaluate(x_test_flat, y_test)[1]
```
:::

::: {.cell .code }
```python
results.append({"model": 'FC Neural Net', "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(11,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }
## Train a convolutional neural network
:::

::: {.cell .code }
```python
filters = 10
model_conv = Sequential()
model_conv.add(InputLayer(input_shape=x_train_img.shape[1:]))
model_conv.add(Conv2D(filters, kernel_size=3, padding="same", activation="relu", use_bias=False ))
model_conv.add(MaxPool2D(pool_size=(2, 2)))
model_conv.add(BatchNormalization())
model_conv.add(Conv2D(filters, kernel_size=3, padding="same", activation="relu", use_bias=False ))
model_conv.add(GlobalAvgPool2D())
model_conv.add(Dense(1, activation="sigmoid"))

model_conv.summary()

model_conv.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
```
:::

::: {.cell .code }
```python
hist = model_conv.fit(x_train_img, y_train, epochs=100, 
     validation_split=0.25,  callbacks=[
        keras.callbacks.ReduceLROnPlateau(factor=.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1)
    ])

train_score = model_conv.evaluate(x_train_img, y_train)[1]
test_score = model_conv.evaluate(x_test_img, y_test)[1]
```
:::

::: {.cell .code }
```python
results.append({"model": 'ConvNet', "train_score": train_score, "test_score": test_score})
```
:::

::: {.cell .code }
```python
results_df = pd.DataFrame(results)

plt.figure(figsize =(12,10));

plt.subplot(2,1,1)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['train_score']);
plt.ylim(0,1);
plt.xlabel("")

plt.subplot(2,1,2)
sns.barplot(x=results_df.sort_values('test_score')['model'], y=results_df.sort_values('test_score')['test_score']);
plt.ylim(0,1);
```
:::

::: {.cell .markdown }
## Using the same model on different slashes
:::

::: {.cell .markdown }
Not only did our convolutional network learn forward and backward
slashes - it can even generalize to slightly different forward and
backward slashes.

Let's generate data with heavier background noise, and longer slashes:
:::

::: {.cell .code }
```python
noise_scale = 0.9
s_pattern = 15
def gen_example_different(size=20, label=0):

    max_s_pattern = int(size // 4)
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,
                                  size=(2, ))  # random x,y point
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] +
       s_pattern] = pattern  # set point to
    ex = ex + noise_scale*(np.random.rand(size, size) - .5)
    np.clip(ex,0.,1., out=ex)
    return ex

examples = []

n_side = 30
n_ex = 50 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example_different(size=n_side, label=0))
    examples.append(gen_example_different(size=n_side, label=1))
    
y_new = np.array([0,1]*n_ex)
x_new = np.stack(examples)

plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y_new), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title(f"y = {y_new[index]}")
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title("yhat =  %0.2f" % model_conv.predict(x_new[index].reshape((1,30,30,1))))
```
:::

::: {.cell .code }
```python
new_test_score = model_conv.evaluate(x_new[...,np.newaxis], y_new)[1]
```
:::

::: {.cell .markdown }
What about forward and backward slashes at different angles?
:::

::: {.cell .code }
```python
max_rot = 10
def gen_example_rotated(size=20, label=0):

    max_s_pattern = int(size // 4)
    s_pattern = 15
    pattern = 1- np.eye(s_pattern)
    if label:
        pattern = pattern[:, ::-1]
    ex = np.ones((size,size))
    point_loc = np.random.randint(0, size - s_pattern + 1,   size=(2, ))  
    ex[point_loc[0]:point_loc[0] + s_pattern, point_loc[1]:point_loc[1] + s_pattern] = pattern  
    rot_angle = np.random.uniform(-max_rot, max_rot)
    ex = scipy.ndimage.rotate(ex, angle=rot_angle, cval=1, reshape = False)
    ex = ex + noise_scale*(np.random.rand(size, size) - .5)

    np.clip(ex,0.,1., out=ex)
    return ex

examples = []

n_side = 30
n_ex = 50 #number of examples in each class

for i in range(n_ex):
    examples.append(gen_example_rotated(size=n_side, label=0))
    examples.append(gen_example_rotated(size=n_side, label=1))
    
y_new = np.array([0,1]*n_ex)
x_new = np.stack(examples)

plt.figure(figsize=(18,4))

n_print = 10 # number of examples to show

ex_indices = np.random.choice(len(y_new), n_print, replace=False)
for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title(f"y = {y_new[index]}")
```
:::

::: {.cell .code }
```python
plt.figure(figsize=(18,4))

for i, index in enumerate(ex_indices):
    plt.subplot(1, n_print, i+1, )
    plt.imshow(x_new[index,...], cmap='gray')
    plt.title("yhat =  %0.2f" % model_conv.predict(x_new[index].reshape((1,30,30,1))))
```
:::

::: {.cell .code }
```python
new_test_score = model_conv.evaluate(x_new[...,np.newaxis], y_new)[1]
```
:::

::: {.cell .markdown }
## Visualizing what the network learns
:::

::: {.cell .code }
```python
from ipywidgets import interactive
from ipywidgets import Layout
import ipywidgets as widgets

def plot_layer(test_idx, layer_idx):
  convout1_f = K.function(model_conv.inputs, [model_conv.layers[layer_idx].output])
  convolutions = np.squeeze(convout1_f(x[test_idx].reshape((1,30,30,1))))
  if (len(convolutions.shape)) > 1:
    m = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(m)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(15,12))
    print(model_conv.layers[layer_idx].name)
    for i in range(m):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[:,:,i], cmap='gray')
        ax.set_title(i)
  else:
    print(model_conv.layers[layer_idx].name)
    plt.imshow(convolutions.reshape(1, convolutions.shape[0]), cmap='gray');
    plt.yticks([])
    plt.xticks(range(convolutions.shape[0]))

style = {'description_width': 'initial'}
layout = Layout(width="800px")
test_idx = widgets.IntSlider(min=0, max=len(x)-1, value=0, style=style, layout=layout)
layer_idx = widgets.IntSlider(min=0, max=len(model_conv.layers)-2, value=0, style=style, layout=layout)
interactive(plot_layer, test_idx=test_idx, layer_idx=layer_idx)
```
:::
